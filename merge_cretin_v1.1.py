#!/usr/bin/env python3
"""
merge_cretin.py  -  FAC detailed data + Screened-Hydrogenic -> Cretin atomic data
==================================================================================

Usage:
    python merge_cretin.py --sh dca_29 --fac Cu_atom_5charge.dat [options]

Options:
    --ion-min  1   Minimum ion_id to process with FAC (Phase 1 default: 1)
    --ion-max  3   Maximum ion_id to process with FAC (Phase 1 default: 3)
    --detail-ions   Ions to keep in FAC detail basis in Phase 2
    --buffer-ions   Ions to keep in FAC same-n average buffer basis in Phase 2
                    (default: automatically choose min/max FAC model ions as buffer)
    --split-degen-tol    Tolerance for same/near-degenerate model energies in Phase 2
                         (default: 1e-9 eV)
    --split-degen-delta  Energy spacing used when splitting degeneracies in Phase 2
                         (default: 1e-3 eV)
    --sampson-source     Source used for Phase 2 sampson ionize
                         fac-colon2(default): FAC colon2 based + SH(sh-only)
                         sh-boundary       : SH boundary remap mode
    --photoionization    phis/phot_ion section handling mode
                         auto(default): include when FAC or SH has photoionization d-lines
                         on          : always include
                         off         : always exclude
    --filter-nonpositive Remove non-positive transition lines in Phase 2 (optional)
    --n-super  4   n >= N uses SH superconfiguration; n < N uses FAC detail
                   (default: 4)
    --phase    1   (default) Safe merge - preserve SH structure and add FAC rates
               2   detail/buffer/SH hybrid basis
    -o OUTPUT      Output filename

Phase 1 - "safe merge" (designed to avoid crashes)
  data model          <- SH as-is (only converted from 10-col -> 22-col)
  data phxs           <- SH as-is
  data phis           <- FAC ions: FAC 8-param / others: SH
  data colex2         <- NEW: FAC ions Burgess-Tully CE
  data colon2         <- NEW: FAC ions CI (BED)
  data sampson excite <- SH as-is
  data sampson ionize <- SH as-is
  data augxs          <- FAC AI rates added + SH
  data augis          <- SH as-is

Phase 2 - "detail / buffer / SH hybrid"
  model:  detail ions -> FAC detail basis
          buffer ions -> FAC same-n averaged basis (1 level per n)
          SH ions     -> SH n-shell basis
  phxs/phis/colex2/colon2/augxs:
          detail ions use FAC detail remap,
          buffer ions regenerate FAC detailed rates as degeneracy-weighted averages
  sampson ionize:
          keep SH-only source behavior, and remap the target by n when the target
          belongs to the managed (detail/buffer) region
"""

import argparse, sys, os, math
from collections import defaultdict


# ── File Parsing ──────────────────────────────────────────────────────────────

def parse_sections(path):
    """Parse the file into sections. Returns: {section_name: [line list]}"""
    sections = {}
    cur_name, cur_lines = None, []
    with open(path) as fh:
        for raw in fh:
            ln = raw.rstrip('\n')
            s  = ln.strip()
            if s.startswith('data '):
                if cur_name is not None:
                    sections[cur_name] = cur_lines
                cur_name  = s[5:].strip()
                cur_lines = [ln]
            elif s == 'end data' and cur_name is not None:
                cur_lines.append(ln)
                sections[cur_name] = cur_lines
                cur_name, cur_lines = None, []
            elif cur_name is not None:
                cur_lines.append(ln)
    if cur_name is not None:
        sections[cur_name] = cur_lines
    return sections


def parse_model_ions(section_lines):
    """
    Parse the model section by ion.
    Returns: {ion_id: {'header': str, 'elev_lines': [str], 'n_to_lev': {n: lev}}}
    """
    ions = {}
    cur = None
    for ln in section_lines:
        s = ln.strip()
        if s.startswith('enot'):
            p = s.split()
            cur = int(p[1])
            ions[cur] = {'header': ln, 'elev_lines': []}
        elif s.startswith('elev') and cur is not None:
            ions[cur]['elev_lines'].append(ln)

    for ion_id, d in ions.items():
        n_to_lev = {}
        for ln in d['elev_lines']:
            s = ln.strip()
            if not s.startswith('elev'):
                continue
            p = s.split()
            if len(p) < 9:          # elev ion lev label deg energy [occ...] n_max
                continue
            lev   = int(p[2])
            n_max = int(p[-1])      # always last token regardless of col count
            if n_max not in n_to_lev:
                n_to_lev[n_max] = lev
        d['n_to_lev'] = n_to_lev
    return ions


def build_fac_lev_nmax(fac_sections, fac_ion_ids):
    """Return {ion_id: {lev: n_max}} from the FAC model."""
    result = {}
    cur = None
    for ln in fac_sections.get('model', []):
        s = ln.strip()
        if s.startswith('enot'):
            cur = int(s.split()[1])
        elif s.startswith('elev') and cur in fac_ion_ids:
            p = s.split()
            if len(p) >= 9:
                lev   = int(p[2])
                n_max = int(p[-1])
                result.setdefault(cur, {})[lev] = n_max
    return result


def build_fac_n_groups(fac_sections, ion_id):
    """Return {n: [sorted lev list]} for ion_id from the FAC model."""
    groups = defaultdict(list)
    cur = None
    for ln in fac_sections.get('model', []):
        s = ln.strip()
        if s.startswith('enot'):
            cur = int(s.split()[1])
        elif s.startswith('elev') and cur == ion_id:
            p = s.split()
            if len(p) >= 9:
                groups[int(p[-1])].append(int(p[2]))
    return {n: sorted(v) for n, v in groups.items()}


# ── elev Format Conversion ────────────────────────────────────────────────────

_MODEL_OCC_COLS = 22


def _detect_model_occ_cols(model_lines, default=22):
    """Infer the number of occupancy columns in model elev lines."""
    for ln in model_lines:
        s = ln.strip()
        if not s.startswith('elev'):
            continue
        p = s.split()
        if len(p) >= 8:
            occ_cols = len(p) - 7
            if occ_cols > 0:
                return occ_cols
    return default


def _set_model_occ_cols(model_lines):
    """Match output elev occupancy column count to the SH model format."""
    global _MODEL_OCC_COLS
    _MODEL_OCC_COLS = _detect_model_occ_cols(model_lines, default=22)


def _to_int_token(tok):
    try:
        return int(float(tok))
    except Exception:
        return 0


def _pad_or_truncate_occ(occ):
    if len(occ) < _MODEL_OCC_COLS:
        return occ + [0] * (_MODEL_OCC_COLS - len(occ))
    return occ[:_MODEL_OCC_COLS]


def _fac_occ_to_22col(cols):
    """
    Normalize FAC occupancy to the model(super configuration) column count.
    - For 36-column FAC, collapse n=4..8 shells
    - Otherwise (e.g. 6-column FAC), keep leading entries and pad with zeros
    """
    c = [_to_int_token(x) for x in cols]

    if len(c) >= 36:
        out = list(c[0:6])
        out.append(sum(c[6:10]))     # n=4
        out.append(sum(c[10:15]))    # n=5
        out.append(sum(c[15:21]))    # n=6
        out.append(sum(c[21:28]))    # n=7
        out.append(sum(c[28:36]))    # n=8
        return _pad_or_truncate_occ(out)

    return _pad_or_truncate_occ(list(c))


def _sh_occ_to_22col_super(cols):
    """
    SH 10-column n-shell occupancy -> model(super) columns.
    n=1,2,3 → pos 0,1,2
    n=k (k≥4) → pos k+2
    """
    c = [_to_int_token(x) for x in cols]
    out = [0] * _MODEL_OCC_COLS
    for i, v in enumerate(c[:10]):
        n = i + 1
        pos = i if n <= 3 else n + 2
        if 0 <= pos < _MODEL_OCC_COLS:
            out[pos] = v
    return out


def _sh_occ_to_22col_nshells(cols):
    """SH 10-column n-shell occupancy -> model(n_shells) columns (direct map + padding)."""
    c = [_to_int_token(x) for x in cols]
    return _pad_or_truncate_occ(c[:10])


def _format_elev(p, occ22, n_max):
    """Format an elev line using normalized occupancy columns."""
    occ_str = '  '.join(str(v) for v in occ22)
    return (f'    elev  {p[1]:>4}  {p[2]:>5}   {p[3]:<16}  '
            f'{p[4]:>8}  {p[5]:>13}    {occ_str}  {n_max}')


def _reformat_fac_elev(ln, new_lev=None):
    """Normalize FAC elev occupancy to the model(super) column width."""
    s = ln.strip()
    p = list(s.split())
    if not s.startswith('elev') or len(p) < 8:
        return ln
    if new_lev is not None:
        p[2] = str(new_lev)
    occ22 = _fac_occ_to_22col(p[6:-1])
    return _format_elev(p, occ22, p[-1])


def _reformat_sh_super_elev(ln, new_lev=None):
    """SH 10-column elev -> 22-column form (c shell section, level renumbering allowed)."""
    s = ln.strip()
    p = list(s.split())
    if not s.startswith('elev') or len(p) < 17:
        return ln
    if new_lev is not None:
        p[2] = str(new_lev)
    occ22 = _sh_occ_to_22col_super(p[6:16])
    return _format_elev(p, occ22, p[16])


def _reformat_sh_nshell_elev(ln, new_lev=None):
    """SH 10-column elev -> 22-column form (c n_shells section)."""
    s = ln.strip()
    p = list(s.split())
    if not s.startswith('elev') or len(p) < 17:
        return ln
    if new_lev is not None:
        p[2] = str(new_lev)
    occ22 = _sh_occ_to_22col_nshells(p[6:16])
    return _format_elev(p, occ22, p[16])


# ── Level Mapping Utilities ───────────────────────────────────────────────────

def _resolve_fac_lev_to(ion_to, old_lev, fac_seq_remap, fac_lev_nmax,
                         lev_remap, sh_n_to_new):
    """
    Map lev_to in FAC phxs/phis (original FAC level number) to the new model level.
    Returns None when the level does not exist in the new model.

    sh_n_to_new[ion_id][n_max] = new_lev  (final FAC+SH merged n->level map)
    """
    # 1) n < n_super: available in fac_seq_remap
    new = fac_seq_remap.get(ion_to, {}).get(old_lev)
    if new is not None:
        return new
    # 2) n >= n_super: find n_max from fac_lev_nmax and apply SH super lev_remap
    n_max = fac_lev_nmax.get(ion_to, {}).get(old_lev)
    if n_max is not None:
        return sh_n_to_new.get(ion_to, {}).get(n_max)
    return None


def _resolve_sh_lev_to(ion_to, sh_lev, fac_ion_ids, lev_remap,
                        sh_n_to_new):
    """
    Map lev_to in SH phis (original SH level number, for a FAC ion) to the new model level.
    Returns None when mapping is not possible.

    sh_n_to_new[ion_id][n_max] = new_lev
    """
    if ion_to not in fac_ion_ids:
        return sh_lev
    # SH super (n >= n_super) -> lev_remap
    new = lev_remap.get(ion_to, {}).get(sh_lev)
    if new is not None:
        return new
    # SH detail (n < n_super) is replaced by FAC detail and should use sh_n_to_new.
    # This path assumes sh_n_to_new also contains the SH level n_max mapping.
    # Here we assume sh_lev itself is the "first lev per n" from SH n_to_lev.
    return None


# ── Section Builders ───────────────────────────────────────────────────────────

def _sec(name, body_lines, comment=None):
    """Wrap a section with header and footer lines."""
    out = [f'data    {name}', '']
    if comment:
        out.append(f'c {comment}')
        out.append('')
    out.extend(body_lines)
    out.append('')
    out.append('end data')
    return out


def sec_model_p1(sh_sections):
    """Phase 1: keep the SH model as-is (no 22-column conversion)."""
    return list(sh_sections.get('model', []))


def sec_model_p2(sh_sections, fac_sections, fac_ion_ids, n_super):
    """
    Phase 2: FAC ions -> FAC detail (n<n_super, sequential renumbering)
             + SH super (n≥n_super)
             SH ions  -> SH n_shells 22-col
    Returns: (lines, lev_remap, fac_seq_remap)
      lev_remap     = {ion_id: {old_sh_lev: new_lev}}     (SH super remap)
      fac_seq_remap = {ion_id: {old_fac_lev: new_seq_lev}} (FAC detail sequential remap)
    """
    sh_ions  = parse_model_ions(sh_sections.get('model', []))
    fac_ions = parse_model_ions(fac_sections.get('model', []))

    body = []
    lev_remap     = {}   # {ion_id: {old_sh_lev: new_lev}}
    fac_seq_remap = {}   # {ion_id: {old_fac_lev: new_seq_lev}}

    for ion_id in sorted(sh_ions.keys()):
        sh_d = sh_ions[ion_id]
        body.append(sh_d['header'])

        if ion_id in fac_ion_ids and ion_id in fac_ions:
            fac_d = fac_ions[ion_id]

            # FAC detail: n < n_super
            detail_lines = [ln for ln in fac_d['elev_lines']
                            if ln.strip().startswith('elev')
                            and int(ln.split()[-1]) < n_super]

            # Renumber FAC levels sequentially starting from 1
            fac_seq_remap[ion_id] = {}
            if detail_lines:
                body.append('c ls_shells')
                for i, ln in enumerate(detail_lines):
                    old_lev = int(ln.split()[2])
                    new_seq = i + 1
                    fac_seq_remap[ion_id][old_lev] = new_seq
                    body.append(_reformat_fac_elev(ln, new_lev=new_seq))

            # SH super: n >= n_super, renumber after FAC detail
            super_lines = [ln for ln in sh_d['elev_lines']
                           if ln.strip().startswith('elev')
                           and int(ln.split()[-1]) >= n_super]

            if super_lines:
                body.append(
                    'c shell    1  2  4  7  8  9 10 11 12 13 14 15 16 17 18 19')
                lev_remap[ion_id] = {}
                start = len(detail_lines) + 1
                for i, ln in enumerate(super_lines):
                    old_lev = int(ln.split()[2])
                    new_lev = start + i
                    lev_remap[ion_id][old_lev] = new_lev
                    body.append(_reformat_sh_super_elev(ln, new_lev=new_lev))

        else:
            # Pure SH ion
            body.append('c n_shells')
            for ln in sh_d['elev_lines']:
                body.append(_reformat_sh_nshell_elev(ln))

        body.append('')

    return _sec('model', body), lev_remap, fac_seq_remap


def sec_phxs_p1(sh_sections):
    return list(sh_sections.get('phxs', []))


def _fmt_fac_phxs(ln, fac_seq_remap):
    """FAC phxs line: sequentially remap level IDs and force 5-param -> 4-param."""
    s = ln.strip()
    p = s.split()
    if not s.startswith('d ') or len(p) < 9:
        return '  ' + '  '.join(p)
    # d ion lev_from ion_to lev_to f lambda [e0 [mix [trailing]]]
    ion_id   = int(p[1])
    old_from = int(p[2])
    ion_to   = int(p[3])
    old_to   = int(p[4])
    new_from = fac_seq_remap.get(ion_id, {}).get(old_from, old_from)
    new_to   = fac_seq_remap.get(ion_to, {}).get(old_to, old_to)
    params   = p[5:9]    # keep at most 4 params (drop the 5th and beyond)
    return (f'  d  {ion_id:3d}  {new_from:5d}  {ion_to:3d}  {new_to:5d}  '
            + '  '.join(params))


def sec_phxs_p2(sh_sections, fac_sections, fac_ion_ids, n_super,
                fac_lev_nmax, lev_remap, fac_seq_remap):
    """
    Build phxs:
      FAC ions, n<n_super  -> FAC 4-param (drop the 5th param, sequential level remap)
      FAC ions, n>=n_super -> SH 3-param (level remap)
      SH ions              -> original SH 3-param entries
    """
    body = []

    # ─ FAC detail phxs (both lev_from and lev_to must be in n<n_super range)
    body.append('c --- FAC detail phxs (4-param)  n < n_super')
    for ln in fac_sections.get('phxs', []):
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = s.split()
        ion_id   = int(p[1])
        ion_to   = int(p[3])
        lev_from = int(p[2])
        lev_to   = int(p[4])
        if ion_id not in fac_ion_ids:
            continue
        nmax_from = fac_lev_nmax.get(ion_id, {}).get(lev_from, 99)
        # ion_to == ion_id for photo-excitation (bound-bound, confirmed)
        nmax_to   = fac_lev_nmax.get(ion_id, {}).get(lev_to, 99)
        if nmax_from < n_super and nmax_to < n_super:
            body.append(_fmt_fac_phxs(ln, fac_seq_remap))

    # SH super phxs for FAC ions (level remap)
    body += ['', 'c --- SH super phxs  n >= n_super  (lev remapped)']
    for ln in sh_sections.get('phxs', []):
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = list(s.split())
        ion_id   = int(p[1])
        lev_from = int(p[2])
        if ion_id not in fac_ion_ids:
            continue
        remap = lev_remap.get(ion_id, {})
        if lev_from not in remap:
            continue
        p[2] = str(remap[lev_from])
        # lev_to (p[4]) is for ion_to; if ion_to is also FAC, remap it too
        ion_to  = int(p[3])
        lev_to  = int(p[4])
        if ion_to in fac_ion_ids:
            remap_to = lev_remap.get(ion_to, {})
            if lev_to in remap_to:
                p[4] = str(remap_to[lev_to])
        body.append('  ' + '  '.join(p))

    # ─ SH phxs for SH ions
    body += ['', 'c --- SH n-shell phxs']
    for ln in sh_sections.get('phxs', []):
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = s.split()
        if int(p[1]) not in fac_ion_ids:
            body.append(ln)

    return _sec('phxs', body)


def sec_phis(sh_sections, fac_sections, fac_ion_ids, n_super=None,
             fac_lev_nmax=None, lev_remap=None, fac_seq_remap=None,
             sh_n_to_new=None):
    """
    Build phis:
      n_super is None (Phase 1): FAC ions use full FAC, SH ions use SH
      Phase 2: FAC detail + SH super (remapped) + SH ions

    sh_n_to_new[ion_id][n_max] = new_model_lev
      Final model-level map for every n_max within a FAC ion.
      n < n_super  -> first FAC detail level
      n >= n_super -> remapped SH super level
    """
    body = []
    fsr  = fac_seq_remap or {}
    snn  = sh_n_to_new   or {}

    if n_super is None:
        # Phase 1: FAC ions -> FAC phis (no level renumbering); SH ions -> SH phis
        fac_phis_source_ions = set()
        for ln in fac_sections.get('phis', []):
            s = ln.strip()
            if not s.startswith('d '):
                continue
            ion_from = int(s.split()[1])
            if ion_from in fac_ion_ids:
                body.append(ln)
                fac_phis_source_ions.add(ion_from)
        # For ions with no FAC phis, fall back to SH phis
        missing_fac_phis_ions = set(fac_ion_ids) - fac_phis_source_ions
        for ln in sh_sections.get('phis', []):
            s = ln.strip()
            if not s.startswith('d '):
                continue
            ion_from = int(s.split()[1])
            if ion_from not in fac_ion_ids or ion_from in missing_fac_phis_ions:
                body.append(ln)
        return _sec('phis', body)

    # Phase 2
    # FAC detail phis (n < n_super)
    for ln in fac_sections.get('phis', []):
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = list(s.split())
        ion_id   = int(p[1])
        lev_from = int(p[2])
        if ion_id not in fac_ion_ids:
            continue
        nmax = fac_lev_nmax.get(ion_id, {}).get(lev_from, 99)
        if nmax >= n_super:
            continue
        # remap lev_from
        p[2] = str(fsr.get(ion_id, {}).get(lev_from, lev_from))
        # remap lev_to (if ion_to is also a FAC ion)
        ion_to = int(p[3])
        lev_to = int(p[4])
        if ion_to in fac_ion_ids:
            new_to = _resolve_fac_lev_to(ion_to, lev_to, fsr,
                                          fac_lev_nmax, lev_remap, snn)
            if new_to is None:
                continue   # lev_to is not present in the model -> skip this transition
            p[4] = str(new_to)
        body.append('  ' + '  '.join(p))

    # SH super phis for FAC ions (n >= n_super, level remap)
    for ln in sh_sections.get('phis', []):
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = list(s.split())
        ion_id   = int(p[1])
        lev_from = int(p[2])
        if ion_id not in fac_ion_ids:
            continue
        remap = lev_remap.get(ion_id, {})
        if lev_from not in remap:
            continue
        p[2] = str(remap[lev_from])
        ion_to = int(p[3])
        lev_to = int(p[4])
        if ion_to in fac_ion_ids:
            # SH lev_to -> new model level: SH super uses lev_remap, SH detail uses sh_n_to_new
            new_to = lev_remap.get(ion_to, {}).get(lev_to)
            if new_to is None:
                # SH detail level (n < n_super): would need a lookup through sh_n_to_new
                # This would require a reverse map from lev_to -> n_max, so skip for now.
                continue
            p[4] = str(new_to)
        body.append('  ' + '  '.join(p))

    # SH phis for SH ions
    for ln in sh_sections.get('phis', []):
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = s.split()
        if int(p[1]) not in fac_ion_ids:
            body.append(ln)

    return _sec('phis', body)


def sec_colex2(fac_sections, fac_ion_ids, n_super=None, fac_lev_nmax=None,
               fac_seq_remap=None):
    """FAC Burgess-Tully CE. Phase 2: only internal n<n_super transitions, with sequential level remap."""
    if 'colex2' not in fac_sections:
        return []
    fsr = fac_seq_remap or {}
    body = []
    for ln in fac_sections['colex2']:
        s = ln.strip()
        if s.startswith('d ') and int(s.split()[1]) in fac_ion_ids:
            p = list(s.split())
            ion_id = int(p[1])
            lev_f  = int(p[2])
            lev_t  = int(p[3])
            if n_super is not None and fac_lev_nmax is not None:
                nmax_f = fac_lev_nmax.get(ion_id, {}).get(lev_f, 99)
                nmax_t = fac_lev_nmax.get(ion_id, {}).get(lev_t, 99)
                if nmax_f >= n_super or nmax_t >= n_super:
                    continue
            p[2] = str(fsr.get(ion_id, {}).get(lev_f, lev_f))
            p[3] = str(fsr.get(ion_id, {}).get(lev_t, lev_t))
            body.append('  ' + '  '.join(p))
        elif s and not s.startswith('data') and not s.startswith('end'):
            body.append(ln)
    return _sec('colex2', body,
                comment='FAC Burgess-Tully CE  ions=' +
                        ','.join(map(str, sorted(fac_ion_ids))))


def sec_colon2(fac_sections, fac_ion_ids, n_super=None, fac_lev_nmax=None,
               fac_seq_remap=None):
    """FAC CI (BED). Phase 2: only internal n<n_super transitions, with sequential level remap."""
    if 'colon2' not in fac_sections:
        return []
    fsr = fac_seq_remap or {}
    body = []
    for ln in fac_sections['colon2']:
        s = ln.strip()
        if s.startswith('d ') and int(s.split()[1]) in fac_ion_ids:
            p = list(s.split())
            ion_id = int(p[1])
            lev_f  = int(p[2])
            lev_t  = int(p[3])
            if n_super is not None and fac_lev_nmax is not None:
                nmax_f = fac_lev_nmax.get(ion_id, {}).get(lev_f, 99)
                nmax_t = fac_lev_nmax.get(ion_id, {}).get(lev_t, 99)
                if nmax_f >= n_super or nmax_t >= n_super:
                    continue
            p[2] = str(fsr.get(ion_id, {}).get(lev_f, lev_f))
            p[3] = str(fsr.get(ion_id, {}).get(lev_t, lev_t))
            body.append('  ' + '  '.join(p))
        elif s and not s.startswith('data') and not s.startswith('end'):
            body.append(ln)
    return _sec('colon2', body,
                comment='FAC CI (BED)  ions=' +
                        ','.join(map(str, sorted(fac_ion_ids))))


def sec_sampson_excite(sh_sections, exclude_ions=None):
    excl = set(exclude_ions or [])
    body = []
    for ln in sh_sections.get('sampson excite', []):
        s = ln.strip()
        if s.startswith('data') or s.startswith('end'):
            continue
        if s.startswith('d ') and int(s.split()[1]) in excl:
            continue
        body.append(ln)
    return _sec('sampson excite', body)


def sec_sampson_excite_p2_hybrid(sh_sections, state):
    body = ['c --- SH supplemental sampson excite (managed+SH remap)']
    body.extend(_build_sh_supplemental_rates(
        sh_sections.get('sampson excite', []), state,
        existing_keys=set(), positive_check=True))
    return _sec('sampson excite', body)


def sec_sampson_ionize_p1(sh_sections):
    body = []
    for ln in sh_sections.get('sampson ionize', []):
        s = ln.strip()
        if s.startswith('data') or s.startswith('end'):
            continue
        body.append(ln)
    return _sec('sampson ionize', body)


def sec_sampson_ionize_p2(sh_sections, sh_ions, fac_n_groups,
                          fac_ion_ids, lev_remap, fac_seq_remap=None):
    """
    Remove FAC-internal transitions and remap cross-boundary level numbers.
    lev_remap:     {ion_id: {old_sh_lev: new_lev}}      - SH super level remap
    fac_seq_remap: {ion_id: {old_fac_lev: new_seq_lev}} - FAC detail sequential remap
    """
    fsr = fac_seq_remap or {}
    body = []
    for ln in sh_sections.get('sampson ionize', []):
        s = ln.strip()
        if s.startswith('data') or s.startswith('end'):
            continue
        if not s.startswith('d '):
            body.append(ln)
            continue

        p = s.split()
        ion_from = int(p[1])
        lev_from = int(p[2])
        ion_to   = int(p[3])
        lev_to   = int(p[4])
        rest     = p[5:]

        # Remove FAC-internal transitions
        if ion_from in fac_ion_ids and ion_to in fac_ion_ids:
            continue

        # Cross-boundary remap when ion_to is a FAC ion
        if ion_to in fac_ion_ids:
            remap = lev_remap.get(ion_to, {})
            if lev_to in remap:
                # SH super level -> new level number
                lev_to = remap[lev_to]
            else:
                # SH detail level -> FAC detail level (by n_max) -> sequential number
                n_to_lev = sh_ions.get(ion_to, {}).get('n_to_lev', {})
                n_val = next((n for n, l in n_to_lev.items()
                              if l == lev_to), None)
                if n_val is not None:
                    fac_levs = fac_n_groups.get(ion_to, {}).get(n_val, [])
                    if fac_levs:
                        old_fac = fac_levs[0]
                        lev_to = fsr.get(ion_to, {}).get(old_fac, old_fac)
                    else:
                        print(f'  [warn] no FAC lev for ion{ion_to} n={n_val}'
                              f', keeping SH lev {lev_to}', file=sys.stderr)
            ln = (f'  d  {ion_from:3d}  {lev_from:5d}  '
                  f'{ion_to:3d}  {lev_to:5d}  ' + '  '.join(rest))

        body.append(ln)
    return _sec('sampson ionize', body)


def sec_augxs(sh_sections, fac_sections, fac_ion_ids,
              n_super=None, fac_lev_nmax=None, fac_seq_remap=None):
    """FAC AI rates (detail levels only, sequentially remapped) + SH AI rates."""
    fsr = fac_seq_remap or {}
    body = []
    if 'augxs' in fac_sections:
        body.append('c --- FAC autoionization rates')
        for ln in fac_sections['augxs']:
            s = ln.strip()
            if s.startswith('d ') and int(s.split()[1]) in fac_ion_ids:
                p = list(s.split())
                ion_id = int(p[1])
                old_lev = int(p[2])
                if n_super is not None and fac_lev_nmax is not None:
                    nmax = fac_lev_nmax.get(ion_id, {}).get(old_lev, 99)
                    if nmax >= n_super:
                        continue
                p[2] = str(fsr.get(ion_id, {}).get(old_lev, old_lev))
                body.append('  ' + '  '.join(p))
        body.append('')
    if 'augxs' in sh_sections:
        body.append('c --- SH autoionization rates')
        for ln in sh_sections['augxs']:
            s = ln.strip()
            if s.startswith('d ') and int(s.split()[1]) not in fac_ion_ids:
                body.append(ln)
    return _sec('augxs', body)


def sec_augis(sh_sections):
    if 'augis' not in sh_sections:
        return []
    body = []
    for ln in sh_sections['augis']:
        s = ln.strip()
        if s.startswith('data') or s.startswith('end'):
            continue
        body.append(ln)
    return _sec('augis', body)


# ── Phase 2 (detail / buffer / SH) ──────────────────────────────────────────

def _safe_float(tok, default=0.0):
    try:
        return float(tok)
    except Exception:
        return default


def _safe_int(tok, default=0):
    try:
        return int(float(tok))
    except Exception:
        return default


def _fmt_sci(val):
    return f'{val:.5E}'


def _build_model_level_tables(model_lines):
    """Convert the model section into per-ion level tables."""
    ions_raw = parse_model_ions(model_lines)
    tables = {}
    for ion_id, d in ions_raw.items():
        levels = []
        lev_to_n = {}
        lev_to_deg = {}
        for ln in d.get('elev_lines', []):
            s = ln.strip()
            p = s.split()
            if not s.startswith('elev') or len(p) < 9:
                continue
            lev = int(p[2])
            n_max = int(p[-1])
            deg = _safe_float(p[4], 1.0)
            levels.append({
                'lev': lev,
                'n': n_max,
                'deg': deg if deg > 0 else 1.0,
                'energy': _safe_float(p[5], 0.0),
                'tokens': p,
                'line': ln,
            })
            lev_to_n[lev] = n_max
            lev_to_deg[lev] = deg if deg > 0 else 1.0
        levels.sort(key=lambda r: r['lev'])
        tables[ion_id] = {
            'header': d.get('header', f'  enot  {ion_id:4d}'),
            'levels': levels,
            'lev_to_n': lev_to_n,
            'lev_to_deg': lev_to_deg,
        }
    return tables


def _fac_level_key(rec):
    """
    FAC buffer grouping key.
    Group by (1s occupancy, n_max, full_occ_signature) so that major
    configurations (for example 3p vs 3d) stay separated within the same n.
    """
    p = rec.get('tokens', [])
    occ_1s = _safe_int(p[6], 0) if len(p) > 6 else 0
    occ_sig = tuple(_safe_int(x, 0) for x in p[6:-1]) if len(p) > 7 else tuple()
    return (occ_1s, rec.get('n', 0), occ_sig)


def _sh_level_key(rec):
    """SH level grouping key used for boundary remap: (1s occupancy, n_max)."""
    p = rec.get('tokens', [])
    occ_1s = _safe_int(p[6], 0) if len(p) > 6 else 0
    return (occ_1s, rec.get('n', 0))


def _occ_tokens_to_nshell_sig(occ_tokens, width=10):
    """
    Project occupancy tokens into an SH n-shell signature of length ``width``.
    - 36-col FAC: l-resolved -> summed into n-shells
    - 6-col FAC : (1s,2s,2p,3s,3p,3d) -> summed into n-shells
    - 10-col SH : used as-is
    """
    c = [_safe_int(x, 0) for x in occ_tokens]
    out = [0] * width

    if len(c) >= 36:
        out[0] = c[0]
        out[1] = c[1] + c[2]
        out[2] = c[3] + c[4] + c[5]
        if width > 3:
            out[3] = sum(c[6:10])
        if width > 4:
            out[4] = sum(c[10:15])
        if width > 5:
            out[5] = sum(c[15:21])
        if width > 6:
            out[6] = sum(c[21:28])
        if width > 7:
            out[7] = sum(c[28:36])
        return tuple(out)

    if len(c) >= 6:
        out[0] = c[0]
        out[1] = c[1] + c[2]
        out[2] = c[3] + c[4] + c[5]
        return tuple(out)

    if len(c) >= 10:
        for i in range(min(width, len(c))):
            out[i] = c[i]
        return tuple(out)

    for i in range(min(width, len(c))):
        out[i] = c[i]
    return tuple(out)


def _select_managed_level_for_sh(state, ion_id, sh_lev):
    """
    Map an SH level to a managed (detail/buffer) level.
    Priority:
      1) candidates with the same n
      2) minimum n-shell occupancy distance
      3) minimum excitation-energy difference
    """
    n_val = state['sh_lev_to_n'].get(ion_id, {}).get(sh_lev)
    candidates = []
    if n_val is not None:
        candidates = list(state['n_to_levels'].get(ion_id, {}).get(n_val, []))
    if not candidates:
        candidates = sorted(state['model_valid_levels'].get(ion_id, set()))
    if not candidates:
        return None

    sh_sig = state['sh_occ_sig'].get(ion_id, {}).get(sh_lev)
    if sh_sig is not None:
        dlev = []
        for lev in candidates:
            sig = state['new_nshell_sig'].get(ion_id, {}).get(lev)
            if sig is None:
                continue
            m = min(len(sig), len(sh_sig))
            d = sum(abs(sig[i] - sh_sig[i]) for i in range(m))
            d += sum(abs(x) for x in sig[m:])
            d += sum(abs(x) for x in sh_sig[m:])
            dlev.append((d, lev))
        if dlev:
            dmin = min(d for d, _ in dlev)
            candidates = [lev for d, lev in dlev if d == dmin]

    sh_e = state['sh_exc'].get(ion_id, {}).get(sh_lev)
    if sh_e is not None:
        candidates = sorted(
            candidates,
            key=lambda lev: (
                abs(state['model_exc'].get(ion_id, {}).get(lev, 1.0e99) - sh_e),
                state['model_exc'].get(ion_id, {}).get(lev, 1.0e99),
                lev))
        return candidates[0]

    return min(candidates)


def _build_buffer_elev_line(ion_id, new_lev, key, fac_group):
    """FAC detail group(key=(1s_occ,n)) -> single averaged elev line."""
    occ_1s = key[0]
    n_val = key[1]
    weights = [max(1.0, rec['deg']) for rec in fac_group]
    wsum = sum(weights)
    eavg = sum(w * rec['energy'] for w, rec in zip(weights, fac_group)) / wsum
    gsum = sum(max(1.0, rec['deg']) for rec in fac_group)

    p0 = fac_group[0]['tokens']
    occ22 = [0] * _MODEL_OCC_COLS
    if len(p0) >= 8:
        occ22 = _fac_occ_to_22col(p0[6:-1])

    fake = ['elev', str(ion_id), str(new_lev), f'1s{occ_1s}_n{n_val}avg',
            f'{gsum:.1f}', f'{eavg:.6f}']
    return _format_elev(fake, occ22, n_val)


def _infer_fac_n_cutoff(fac_levels, n_super):
    """
    Estimate the highest n that is explicitly covered in detail by the FAC input.
    If n_super is given, cap the cutoff there so higher n can be handed to the SH tail.
    """
    if not fac_levels:
        return 0
    fac_max_n = max(rec.get('n', 0) for rec in fac_levels)
    if n_super is None:
        return fac_max_n
    return min(fac_max_n, max(0, n_super - 1))


def sec_model_p2_hybrid(sh_sections, fac_sections, detail_ions, buffer_ions, n_super):
    """
    Phase 2 hybrid model:
      detail ion: pure FAC detail basis
      buffer ion: FAC averaged low-n basis + SH super tail (n >= n_super)
      SH ion    : SH n-shell basis
    """
    sh_tbl = _build_model_level_tables(sh_sections.get('model', []))
    fac_tbl = _build_model_level_tables(fac_sections.get('model', []))

    detail_req = set(detail_ions or [])
    buffer_req = set(buffer_ions or [])
    overlap = detail_req & buffer_req
    if overlap:
        print(f'  [warn] detail/buffer overlap={sorted(overlap)}; '
              f'overlap ions are forced to detail', file=sys.stderr)
    buffer_req -= detail_req

    missing_detail = sorted(i for i in detail_req if i not in fac_tbl)
    missing_buffer = sorted(i for i in buffer_req if i not in fac_tbl)
    if missing_detail:
        print(f'  [warn] detail ions missing in FAC model: {missing_detail}',
              file=sys.stderr)
    if missing_buffer:
        print(f'  [warn] buffer ions missing in FAC model: {missing_buffer}',
              file=sys.stderr)

    state = {
        'policy': {},
        'managed_ions': set(),
        'detail_ions': set(),
        'buffer_ions': set(),
        'fac_to_new': defaultdict(dict),   # FAC old lev -> new model lev
        'sh_to_new': defaultdict(dict),    # SH old lev -> new model lev (managed ions)
        'n_to_new': defaultdict(dict),     # n_max -> new model lev (managed ions)
        'n_to_levels': defaultdict(lambda: defaultdict(list)),   # n_max -> [new lev]
        'key_to_new': defaultdict(dict),   # FAC key -> new model lev (managed ions)
        'new_nshell_sig': defaultdict(dict),   # new model lev -> SH n-shell signature
        'fac_key': defaultdict(dict),      # FAC old lev -> (1s_occ,n)
        'sh_key': defaultdict(dict),       # SH old lev -> (1s_occ,n)
        'sh_deg': defaultdict(dict),       # SH old lev degeneracy
        'sh_occ_sig': defaultdict(dict),   # SH old lev -> SH n-shell signature
        'sh_exc': defaultdict(dict),       # SH old lev -> excitation energy
        'fac_groups': defaultdict(dict),   # new model lev -> [old FAC lev]
        'fac_deg': defaultdict(dict),      # FAC old lev degeneracy
        'fac_nmax': defaultdict(dict),     # FAC old lev n_max
        'fac_n_cutoff': {},
        'sh_lev_to_n': defaultdict(dict),  # SH old lev -> n_max
        'model_exc': defaultdict(dict),    # model excitation energy by (ion,lev)
        'model_valid_levels': defaultdict(set),
    }

    body = []
    for ion_id in sorted(sh_tbl.keys()):
        sh_d = sh_tbl[ion_id]

        policy = 'sh'
        if ion_id in detail_req and ion_id in fac_tbl:
            policy = 'detail'
        elif ion_id in buffer_req and ion_id in fac_tbl:
            policy = 'buffer'
        state['policy'][ion_id] = policy

        for rec in sh_d['levels']:
            state['sh_lev_to_n'][ion_id][rec['lev']] = rec['n']
            state['sh_key'][ion_id][rec['lev']] = _sh_level_key(rec)
            state['sh_deg'][ion_id][rec['lev']] = rec['deg']
            state['sh_occ_sig'][ion_id][rec['lev']] = _occ_tokens_to_nshell_sig(
                rec['tokens'][6:-1] if len(rec['tokens']) > 7 else [])
            state['sh_exc'][ion_id][rec['lev']] = rec['energy']

        if policy == 'detail':
            fac_d = fac_tbl[ion_id]
            state['managed_ions'].add(ion_id)
            state['detail_ions'].add(ion_id)
            fac_n_cut = _infer_fac_n_cutoff(fac_d['levels'], n_super)
            state['fac_n_cutoff'][ion_id] = fac_n_cut

            # Keep the absolute ground reference (enot) anchored to the SH backbone
            body.append(sh_d['header'])
            body.append('c ls_shells')

            new_lev = 1
            for rec in fac_d['levels']:
                if rec['n'] > fac_n_cut:
                    continue
                old = rec['lev']
                n_val = rec['n']
                key = _fac_level_key(rec)
                state['fac_to_new'][ion_id][old] = new_lev
                state['fac_key'][ion_id][old] = key
                state['fac_groups'][ion_id][new_lev] = [old]
                state['fac_deg'][ion_id][old] = rec['deg']
                state['fac_nmax'][ion_id][old] = n_val
                state['model_exc'][ion_id][new_lev] = rec['energy']
                state['model_valid_levels'][ion_id].add(new_lev)
                state['new_nshell_sig'][ion_id][new_lev] = _occ_tokens_to_nshell_sig(
                    rec['tokens'][6:-1] if len(rec['tokens']) > 7 else [])
                state['n_to_levels'][ion_id][n_val].append(new_lev)
                if n_val not in state['n_to_new'][ion_id]:
                    state['n_to_new'][ion_id][n_val] = new_lev
                if key not in state['key_to_new'][ion_id]:
                    state['key_to_new'][ion_id][key] = new_lev
                body.append(_reformat_fac_elev(rec['line'], new_lev=new_lev))
                new_lev += 1

            sh_direct = {}
            for sh_rec in sh_d['levels']:
                if sh_rec['n'] <= fac_n_cut:
                    continue
                sh_direct[sh_rec['lev']] = new_lev
                state['model_exc'][ion_id][new_lev] = sh_rec['energy']
                state['model_valid_levels'][ion_id].add(new_lev)
                state['new_nshell_sig'][ion_id][new_lev] = (
                    state['sh_occ_sig'].get(ion_id, {}).get(sh_rec['lev']))
                state['n_to_levels'][ion_id][sh_rec['n']].append(new_lev)
                if sh_rec['n'] not in state['n_to_new'][ion_id]:
                    state['n_to_new'][ion_id][sh_rec['n']] = new_lev
                body.append(_reformat_sh_nshell_elev(sh_rec['line'], new_lev=new_lev))
                new_lev += 1

            for sh_rec in sh_d['levels']:
                direct = sh_direct.get(sh_rec['lev'])
                if direct is not None:
                    state['sh_to_new'][ion_id][sh_rec['lev']] = direct
                    continue
                mapped = _select_managed_level_for_sh(state, ion_id, sh_rec['lev'])
                if mapped is not None:
                    state['sh_to_new'][ion_id][sh_rec['lev']] = mapped
            body.append('')
            continue

        if policy == 'buffer':
            fac_d = fac_tbl[ion_id]
            state['managed_ions'].add(ion_id)
            state['buffer_ions'].add(ion_id)
            fac_n_cut = _infer_fac_n_cutoff(fac_d['levels'], n_super)
            state['fac_n_cutoff'][ion_id] = fac_n_cut

            # Keep the absolute ground reference (enot) anchored to the SH backbone
            body.append(sh_d['header'])
            body.append('c n_shells')

            groups = defaultdict(list)
            for rec in fac_d['levels']:
                if rec['n'] > fac_n_cut:
                    continue
                key = _fac_level_key(rec)
                groups[key].append(rec)

            new_lev = 1
            sort_items = []
            for key, grp0 in groups.items():
                grp_sorted = sorted(grp0, key=lambda r: r['lev'])
                weights0 = [max(1.0, r['deg']) for r in grp_sorted]
                wsum0 = sum(weights0)
                eavg0 = sum(w * r['energy'] for w, r in zip(weights0, grp_sorted)) / wsum0
                sort_items.append((eavg0, key[1], key, grp_sorted))

            for _eavg_sort, n_val, key, grp in sorted(sort_items, key=lambda x: (x[0], x[1], x[2])):
                old_levs = [r['lev'] for r in grp]
                weights = [max(1.0, r['deg']) for r in grp]
                wsum = sum(weights)
                eavg = sum(w * r['energy'] for w, r in zip(weights, grp)) / wsum

                for old in old_levs:
                    state['fac_to_new'][ion_id][old] = new_lev
                    state['fac_key'][ion_id][old] = key
                    state['fac_deg'][ion_id][old] = next(
                        (r['deg'] for r in grp if r['lev'] == old), 1.0)
                    state['fac_nmax'][ion_id][old] = n_val
                state['fac_groups'][ion_id][new_lev] = old_levs
                state['new_nshell_sig'][ion_id][new_lev] = _occ_tokens_to_nshell_sig(
                    grp[0]['tokens'][6:-1] if grp and len(grp[0]['tokens']) > 7 else [])
                state['n_to_levels'][ion_id][n_val].append(new_lev)
                if n_val not in state['n_to_new'][ion_id]:
                    state['n_to_new'][ion_id][n_val] = new_lev
                state['key_to_new'][ion_id][key] = new_lev
                state['model_exc'][ion_id][new_lev] = eavg
                state['model_valid_levels'][ion_id].add(new_lev)
                body.append(_build_buffer_elev_line(ion_id, new_lev, key, grp))
                new_lev += 1

            # Preserve the SH superconfiguration high-n / inner-hole ladder
            # for buffer ions so the boundary-ionization target manifold remains available.
            sh_direct = {}
            for sh_rec in sh_d['levels']:
                if sh_rec['n'] <= fac_n_cut:
                    continue
                sh_direct[sh_rec['lev']] = new_lev
                state['model_exc'][ion_id][new_lev] = sh_rec['energy']
                state['model_valid_levels'][ion_id].add(new_lev)
                state['new_nshell_sig'][ion_id][new_lev] = (
                    state['sh_occ_sig'].get(ion_id, {}).get(sh_rec['lev']))
                state['n_to_levels'][ion_id][sh_rec['n']].append(new_lev)
                if sh_rec['n'] not in state['n_to_new'][ion_id]:
                    state['n_to_new'][ion_id][sh_rec['n']] = new_lev
                body.append(_reformat_sh_nshell_elev(sh_rec['line'], new_lev=new_lev))
                new_lev += 1

            for sh_rec in sh_d['levels']:
                direct = sh_direct.get(sh_rec['lev'])
                if direct is not None:
                    state['sh_to_new'][ion_id][sh_rec['lev']] = direct
                    continue
                mapped = _select_managed_level_for_sh(state, ion_id, sh_rec['lev'])
                if mapped is not None:
                    state['sh_to_new'][ion_id][sh_rec['lev']] = mapped
            body.append('')
            continue

        body.append(sh_d['header'])
        body.append('c n_shells')
        for rec in sh_d['levels']:
            state['model_exc'][ion_id][rec['lev']] = rec['energy']
            state['model_valid_levels'][ion_id].add(rec['lev'])
            body.append(_reformat_sh_nshell_elev(rec['line']))
        body.append('')

    return _sec('model', body), state


def _format_rate_line(ion_from, lev_from, ion_to, lev_to, param_tokens):
    if param_tokens:
        return (f'  d  {ion_from:3d}  {lev_from:5d}  {ion_to:3d}  {lev_to:5d}  '
                + '  '.join(param_tokens))
    return f'  d  {ion_from:3d}  {lev_from:5d}  {ion_to:3d}  {lev_to:5d}'


def _map_fac_level_for_model(state, ion_id, fac_lev):
    return state['fac_to_new'].get(ion_id, {}).get(fac_lev)


def _fallback_key_map(state, ion_id, sh_key, sh_lev=None):
    """Fall back to the nearest managed key when the exact key is unavailable."""
    if sh_lev is not None:
        mapped = _select_managed_level_for_sh(state, ion_id, sh_lev)
        if mapped is not None:
            return mapped

    if sh_key is None:
        return None
    key_to_new = state['key_to_new'].get(ion_id, {})
    if not key_to_new:
        return None
    if sh_key in key_to_new:
        return key_to_new[sh_key]

    occ, n_val = sh_key
    # 1) Prefer the smallest n>=n_val with the same 1s occupancy; otherwise the nearest n with the same occupancy
    same_occ = [((k[1] - n_val) if k[1] >= n_val else 10**6 + abs(k[1] - n_val),
                 abs(k[1] - n_val), k[1], lev)
                for k, lev in key_to_new.items() if k[0] == occ]
    if same_occ:
        same_occ.sort()
        return same_occ[0][3]

    # 2) Within the same n, prefer the closest occupancy
    same_n = [(abs(k[0] - occ), k[0], lev)
              for k, lev in key_to_new.items() if k[1] == n_val]
    if same_n:
        same_n.sort()
        return same_n[0][2]

    return None


def _map_sh_level_for_model(state, ion_id, sh_lev):
    if ion_id in state['managed_ions']:
        mapped = state['sh_to_new'].get(ion_id, {}).get(sh_lev)
        if mapped is not None:
            return mapped
        sh_key = state['sh_key'].get(ion_id, {}).get(sh_lev)
        mapped = _fallback_key_map(state, ion_id, sh_key, sh_lev=sh_lev)
        if mapped is not None:
            return mapped
        n_val = state['sh_lev_to_n'].get(ion_id, {}).get(sh_lev)
        if n_val is not None:
            cands = state['n_to_levels'].get(ion_id, {}).get(n_val, [])
            if cands:
                return min(cands)
        return None
    return sh_lev


def _transition_is_positive_exc(state, ion_from, lev_from, ion_to, lev_to, eps=1e-12):
    ef = state['model_exc'].get(ion_from, {}).get(lev_from)
    et = state['model_exc'].get(ion_to, {}).get(lev_to)
    if ef is None or et is None:
        return True
    return et > ef + eps


def _acc_weighted(agg, key, params, weight):
    if key not in agg:
        agg[key] = {'w': 0.0, 'v': [0.0] * len(params)}
    agg[key]['w'] += weight
    for i, val in enumerate(params):
        agg[key]['v'][i] += weight * val


def _emit_weighted_lines(agg):
    lines = []
    for key in sorted(agg.keys()):
        ion_from, lev_from, ion_to, lev_to = key
        w = agg[key]['w']
        if w <= 0:
            continue
        vals = [v / w for v in agg[key]['v']]
        lines.append(_format_rate_line(
            ion_from, lev_from, ion_to, lev_to, [_fmt_sci(v) for v in vals]))
    return lines


def _emit_weighted_lines_lenkey(agg):
    """Aggregator whose key is (ion_from, lev_from, ion_to, lev_to, nparam)."""
    lines = []
    for key in sorted(agg.keys()):
        ion_from, lev_from, ion_to, lev_to, _nparam = key
        w = agg[key]['w']
        if w <= 0:
            continue
        vals = [v / w for v in agg[key]['v']]
        lines.append(_format_rate_line(
            ion_from, lev_from, ion_to, lev_to, [_fmt_sci(v) for v in vals]))
    return lines


def _collect_transition_keys(lines):
    keys = set()
    for ln in lines:
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = s.split()
        if len(p) < 5:
            continue
        keys.add((int(p[1]), int(p[2]), int(p[3]), int(p[4])))
    return keys


def _build_sh_supplemental_rates(section_lines, state, existing_keys=None,
                                 positive_check=False):
    """
    Remap SH rates onto the current model levels and build supplemental lines.
    - Do not add transitions already present in existing_keys (FAC takes precedence)
    - If remap collisions fold multiple SH lines into one new transition,
      use a source-degeneracy-weighted average
    """
    existing = set(existing_keys or set())
    agg = {}
    passthrough = []

    for ln in section_lines:
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = s.split()
        if len(p) < 5:
            continue

        ion_from_old = int(p[1])
        lev_from_old = int(p[2])
        ion_to_old = int(p[3])
        lev_to_old = int(p[4])

        remap_from = ion_from_old in state['managed_ions']
        remap_to = ion_to_old in state['managed_ions']

        ion_from = ion_from_old
        lev_from = lev_from_old
        if remap_from:
            lev_from = _map_sh_level_for_model(state, ion_from, lev_from_old)
            if lev_from is None:
                continue

        ion_to = ion_to_old
        lev_to = lev_to_old
        if remap_to:
            lev_to = _map_sh_level_for_model(state, ion_to, lev_to_old)
            if lev_to is None:
                continue

        if positive_check and not _transition_is_positive_exc(
                state, ion_from, lev_from, ion_to, lev_to):
            continue

        param_tokens = p[5:]
        if not param_tokens:
            continue

        # Preserve SH-only lines exactly as they are to avoid unnecessary averaging.
        if not remap_from and not remap_to:
            passthrough.append(_format_rate_line(
                ion_from_old, lev_from_old, ion_to_old, lev_to_old, param_tokens))
            continue

        tkey = (ion_from, lev_from, ion_to, lev_to)
        if tkey in existing:
            continue

        params = [_safe_float(x) for x in param_tokens]
        w = state['sh_deg'].get(ion_from_old, {}).get(lev_from_old, 1.0)
        _acc_weighted(agg, tkey + (len(params),), params, w)

    lines = passthrough + _emit_weighted_lines_lenkey(agg)
    return lines


def _photo_level_weight(state, ion_id, lev):
    """
    Level weights for photoionization distribution.
    Use only statistical weight (g) to avoid introducing temperature dependence
    at the atomic-data stage.
    """
    g = 1.0
    if ion_id in state['managed_ions']:
        olds = state['fac_groups'].get(ion_id, {}).get(lev, [])
        if olds:
            g = sum(max(1.0, state['fac_deg'].get(ion_id, {}).get(old, 1.0))
                    for old in olds)
    return max(1.0e-30, g)


def _build_sh_photo_supplemental_rates(section_lines, state, existing_keys=None,
                                       fac_source_ions=None):
    """
    SH photoionization supplementation:
    - For managed ions with FAC photo data, use the standard remap supplement
      (_build_sh_supplemental_rates)
    - For managed ions without FAC photo data, distribute the SH channel total
      over FAC levels while preserving total p1 for each
      (ion_from, ion_to, lev_to, nparam)
    """
    existing = set(existing_keys or set())
    fac_src = set(fac_source_ions or set())
    managed_missing = set(state['managed_ions']) - fac_src

    passthrough_lines = []
    dist_input = []
    for ln in section_lines:
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = s.split()
        if len(p) < 6:
            continue
        ion_from_old = int(p[1])
        if ion_from_old in managed_missing:
            dist_input.append(ln)
        else:
            passthrough_lines.append(ln)

    out_lines = _build_sh_supplemental_rates(
        passthrough_lines, state, existing_keys=existing, positive_check=True)
    existing.update(_collect_transition_keys(out_lines))

    # For managed ions missing FAC photo data: distribute SH totals onto FAC levels
    groups = defaultdict(list)
    for ln in dist_input:
        s = ln.strip()
        p = s.split()
        if len(p) < 6:
            continue
        ion_from_old = int(p[1])
        ion_to_old = int(p[3])
        lev_to_old = int(p[4])

        ion_from = ion_from_old
        ion_to = ion_to_old
        lev_to = lev_to_old
        if ion_to in state['managed_ions']:
            lev_to = _map_sh_level_for_model(state, ion_to, lev_to_old)
            if lev_to is None:
                continue

        params = [_safe_float(x) for x in p[5:]]
        if not params:
            continue
        nparam = len(params)
        groups[(ion_from, ion_to, lev_to, nparam)].append(params)

    for key in sorted(groups.keys()):
        ion_from, ion_to, lev_to, nparam = key
        model_levels = sorted(state['model_valid_levels'].get(ion_from, set()))
        if not model_levels:
            continue

        rows = groups[key]
        total_p1 = sum(max(0.0, r[0]) for r in rows)
        if total_p1 <= 0.0:
            total_p1 = sum(r[0] for r in rows)
        if total_p1 <= 0.0:
            continue

        # Use p1 as the weight when averaging the remaining parameters to preserve spectral shape
        p1w = sum(max(0.0, r[0]) for r in rows)
        if p1w <= 0.0:
            p1w = float(len(rows))
            tail = [sum(r[i] for r in rows) / p1w for i in range(1, nparam)]
        else:
            tail = [sum(max(0.0, r[0]) * r[i] for r in rows) / p1w
                    for i in range(1, nparam)]

        weighted_levels = []
        for lev_from in model_levels:
            if ion_from != ion_to and not _transition_is_positive_exc(
                    state, ion_from, lev_from, ion_to, lev_to):
                continue
            tkey = (ion_from, lev_from, ion_to, lev_to)
            if tkey in existing:
                continue
            w = _photo_level_weight(state, ion_from, lev_from)
            weighted_levels.append((lev_from, w))

        if not weighted_levels:
            continue

        wsum = sum(max(0.0, w) for _, w in weighted_levels)
        if wsum <= 0.0:
            wsum = float(len(weighted_levels))
            weighted_levels = [(lev, 1.0) for lev, _ in weighted_levels]

        for lev_from, w in weighted_levels:
            frac = max(0.0, w) / wsum
            p1 = total_p1 * frac
            params = [_fmt_sci(p1)] + [_fmt_sci(v) for v in tail]
            out_lines.append(_format_rate_line(
                ion_from, lev_from, ion_to, lev_to, params))
            existing.add((ion_from, lev_from, ion_to, lev_to))

    return out_lines


def sec_phxs_p2_hybrid(sh_sections, fac_sections, state):
    fac_body = []

    fac_body.append('c --- FAC detail phxs')
    for ln in fac_sections.get('phxs', []):
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = s.split()
        ion_from = int(p[1])
        if ion_from not in state['detail_ions']:
            continue
        lev_from_old = int(p[2])
        ion_to = int(p[3])
        lev_to_old = int(p[4])
        lev_from = _map_fac_level_for_model(state, ion_from, lev_from_old)
        if lev_from is None:
            continue
        lev_to = lev_to_old
        if ion_to in state['managed_ions']:
            lev_to = _map_fac_level_for_model(state, ion_to, lev_to_old)
            if lev_to is None:
                continue
        fac_body.append(_format_rate_line(ion_from, lev_from, ion_to, lev_to, p[5:9]))

    fac_body.append('')
    fac_body.append('c --- FAC buffer phxs (same-n averaged)')
    agg = {}
    for ln in fac_sections.get('phxs', []):
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = s.split()
        ion_from = int(p[1])
        if ion_from not in state['buffer_ions']:
            continue
        lev_from_old = int(p[2])
        ion_to = int(p[3])
        lev_to_old = int(p[4])
        lev_from = _map_fac_level_for_model(state, ion_from, lev_from_old)
        if lev_from is None:
            continue
        lev_to = lev_to_old
        if ion_to in state['managed_ions']:
            lev_to = _map_fac_level_for_model(state, ion_to, lev_to_old)
            if lev_to is None:
                continue
        params = [_safe_float(x) for x in p[5:9]]
        w = state['fac_deg'].get(ion_from, {}).get(lev_from_old, 1.0)
        _acc_weighted(agg, (ion_from, lev_from, ion_to, lev_to), params, w)
    fac_body.extend(_emit_weighted_lines(agg))

    fac_existing = _collect_transition_keys(fac_body)
    sh_lines = _build_sh_supplemental_rates(
        sh_sections.get('phxs', []), state, existing_keys=fac_existing,
        positive_check=True)

    body = []
    body.append('c --- SH supplemental phxs (managed+SH remap, FAC precedence)')
    body.extend(sh_lines)
    body.append('')
    body.extend(fac_body)

    return _sec('phxs', body)


def sec_phis_p2_hybrid(sh_sections, fac_sections, state):
    body = []
    seen_t = set()
    fac_source_ions = set()

    def _append_unique_transition(ion_from, lev_from, ion_to, lev_to, params):
        tkey = (ion_from, lev_from, ion_to, lev_to)
        if tkey in seen_t:
            return
        seen_t.add(tkey)
        body.append(_format_rate_line(ion_from, lev_from, ion_to, lev_to, params))

    body.append('c --- FAC detail phis')
    for ln in fac_sections.get('phis', []):
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = s.split()
        ion_from = int(p[1])
        if ion_from not in state['detail_ions']:
            continue
        lev_from_old = int(p[2])
        ion_to = int(p[3])
        lev_to_old = int(p[4])
        lev_from = _map_fac_level_for_model(state, ion_from, lev_from_old)
        if lev_from is None:
            continue
        lev_to = lev_to_old
        if ion_to in state['managed_ions']:
            lev_to = _map_fac_level_for_model(state, ion_to, lev_to_old)
            if lev_to is None:
                continue
        if ion_from != ion_to and not _transition_is_positive_exc(
                state, ion_from, lev_from, ion_to, lev_to):
            continue
        fac_source_ions.add(ion_from)
        _append_unique_transition(ion_from, lev_from, ion_to, lev_to, p[5:])

    body.append('')
    body.append('c --- FAC buffer phis (same-n averaged)')
    agg = {}
    for ln in fac_sections.get('phis', []):
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = s.split()
        ion_from = int(p[1])
        if ion_from not in state['buffer_ions']:
            continue
        lev_from_old = int(p[2])
        ion_to = int(p[3])
        lev_to_old = int(p[4])
        lev_from = _map_fac_level_for_model(state, ion_from, lev_from_old)
        if lev_from is None:
            continue
        lev_to = lev_to_old
        if ion_to in state['managed_ions']:
            lev_to = _map_fac_level_for_model(state, ion_to, lev_to_old)
            if lev_to is None:
                continue
        if ion_from != ion_to and not _transition_is_positive_exc(
                state, ion_from, lev_from, ion_to, lev_to):
            continue
        fac_source_ions.add(ion_from)
        params = [_safe_float(x) for x in p[5:]]
        w = state['fac_deg'].get(ion_from, {}).get(lev_from_old, 1.0)
        _acc_weighted(agg, (ion_from, lev_from, ion_to, lev_to), params, w)
    for key in sorted(agg.keys()):
        ion_from, lev_from, ion_to, lev_to = key
        if (ion_from, lev_from, ion_to, lev_to) in seen_t:
            continue
        w = agg[key]['w']
        if w <= 0:
            continue
        vals = [v / w for v in agg[key]['v']]
        _append_unique_transition(
            ion_from, lev_from, ion_to, lev_to, [_fmt_sci(v) for v in vals])

    body.append('')
    body.append('c --- SH supplemental phis (managed remap/distribution, FAC precedence)')
    body.extend(_build_sh_photo_supplemental_rates(
        sh_sections.get('phis', []), state, existing_keys=seen_t,
        fac_source_ions=fac_source_ions))

    return _sec('phis', body)


def _sec_collisional_p2_hybrid(sec_name, fac_sections, state):
    if sec_name not in fac_sections:
        return []
    body = []

    body.append(f'c --- FAC detail {sec_name}')
    for ln in fac_sections.get(sec_name, []):
        s = ln.strip()
        if not s.startswith('d '):
            if s and not s.startswith('data') and not s.startswith('end'):
                body.append(ln)
            continue
        p = s.split()
        ion_from = int(p[1])
        if ion_from not in state['detail_ions']:
            continue
        lev_from_old = int(p[2])
        ion_to = int(p[3])
        lev_to_old = int(p[4])
        lev_from = _map_fac_level_for_model(state, ion_from, lev_from_old)
        if lev_from is None:
            continue
        lev_to = lev_to_old
        if ion_to in state['managed_ions']:
            lev_to = _map_fac_level_for_model(state, ion_to, lev_to_old)
            if lev_to is None:
                continue
        if ion_from != ion_to and not _transition_is_positive_exc(
                state, ion_from, lev_from, ion_to, lev_to):
            continue
        body.append(_format_rate_line(ion_from, lev_from, ion_to, lev_to, p[5:]))

    body.append('')
    body.append(f'c --- FAC buffer {sec_name} (same-n averaged)')
    agg = {}
    for ln in fac_sections.get(sec_name, []):
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = s.split()
        ion_from = int(p[1])
        if ion_from not in state['buffer_ions']:
            continue
        lev_from_old = int(p[2])
        ion_to = int(p[3])
        lev_to_old = int(p[4])
        lev_from = _map_fac_level_for_model(state, ion_from, lev_from_old)
        if lev_from is None:
            continue
        lev_to = lev_to_old
        if ion_to in state['managed_ions']:
            lev_to = _map_fac_level_for_model(state, ion_to, lev_to_old)
            if lev_to is None:
                continue
        if ion_from != ion_to and not _transition_is_positive_exc(
                state, ion_from, lev_from, ion_to, lev_to):
            continue
        params = [_safe_float(x) for x in p[5:]]
        w = state['fac_deg'].get(ion_from, {}).get(lev_from_old, 1.0)
        _acc_weighted(agg, (ion_from, lev_from, ion_to, lev_to), params, w)
    body.extend(_emit_weighted_lines(agg))

    return _sec(sec_name, body)


def sec_augxs_p2_hybrid(sh_sections, fac_sections, state):
    fac_body = []

    if 'augxs' in fac_sections:
        fac_body.append('c --- FAC augxs (detail)')
        for ln in fac_sections['augxs']:
            s = ln.strip()
            if not s.startswith('d '):
                continue
            p = s.split()
            ion_from = int(p[1])
            if ion_from not in state['detail_ions']:
                continue
            lev_from_old = int(p[2])
            ion_to = int(p[3])
            lev_to_old = int(p[4])
            lev_from = _map_fac_level_for_model(state, ion_from, lev_from_old)
            if lev_from is None:
                continue
            lev_to = lev_to_old
            if ion_to in state['managed_ions']:
                lev_to = _map_fac_level_for_model(state, ion_to, lev_to_old)
                if lev_to is None:
                    continue
            fac_body.append(_format_rate_line(
                ion_from, lev_from, ion_to, lev_to, p[5:]))

        fac_body.append('')
        fac_body.append('c --- FAC augxs (buffer averaged)')
        agg = {}
        for ln in fac_sections['augxs']:
            s = ln.strip()
            if not s.startswith('d '):
                continue
            p = s.split()
            ion_from = int(p[1])
            if ion_from not in state['buffer_ions']:
                continue
            lev_from_old = int(p[2])
            ion_to = int(p[3])
            lev_to_old = int(p[4])
            lev_from = _map_fac_level_for_model(state, ion_from, lev_from_old)
            if lev_from is None:
                continue
            lev_to = lev_to_old
            if ion_to in state['managed_ions']:
                lev_to = _map_fac_level_for_model(state, ion_to, lev_to_old)
                if lev_to is None:
                    continue
            params = [_safe_float(x) for x in p[5:]]
            w = state['fac_deg'].get(ion_from, {}).get(lev_from_old, 1.0)
            _acc_weighted(agg, (ion_from, lev_from, ion_to, lev_to), params, w)
        fac_body.extend(_emit_weighted_lines(agg))
        fac_body.append('')

    sh_lines = []
    if 'augxs' in sh_sections:
        existing = _collect_transition_keys(fac_body)
        sh_lines.append('c --- SH supplemental augxs (managed+SH remap, FAC precedence)')
        sh_lines.extend(_build_sh_supplemental_rates(
            sh_sections['augxs'], state, existing_keys=existing,
            positive_check=False))

    body = []
    if sh_lines:
        body.extend(sh_lines)
        if fac_body:
            body.append('')
    body.extend(fac_body)

    return _sec('augxs', body)


def sec_sampson_ionize_p2_hybrid(sh_sections, state):
    """Keep SH sampson ionize, but remap only the boundary cases whose target is a managed ion."""
    body = []
    for ln in sh_sections.get('sampson ionize', []):
        s = ln.strip()
        if s.startswith('data') or s.startswith('end'):
            continue
        if not s.startswith('d '):
            body.append(ln)
            continue

        p = s.split()
        ion_from = int(p[1])
        lev_from = int(p[2])
        ion_to = int(p[3])
        lev_to = int(p[4])

        if ion_from in state['managed_ions']:
            continue

        if ion_to in state['managed_ions']:
            mapped = _map_sh_level_for_model(state, ion_to, lev_to)
            if mapped is None:
                n_val = state['sh_lev_to_n'].get(ion_to, {}).get(lev_to)
                print(f'  [warn] sampson ionize skip: ion_to={ion_to} '
                      f'sh_lev={lev_to} n={n_val}', file=sys.stderr)
                continue
            lev_to = mapped
        body.append(_format_rate_line(ion_from, lev_from, ion_to, lev_to, p[5:]))
    return _sec('sampson ionize', body)


def sec_sampson_ionize_from_fac_colon2_p2(sh_sections, fac_sections, state):
    """
    Build Phase-2 sampson ionize from FAC colon2.
    - For managed-ion sources (detail/buffer), use remapped/averaged FAC colon2
    - For SH-only regions (both source and target in SH), keep the original SH sampson ionize
    - In other words, the section label remains sampson ionize, but the managed/detail-buffer
      rates themselves are FAC-based hybrid values. SH remains only for non-managed
      supplementation and connectivity.
    """
    fac_body = []

    if 'colon2' in fac_sections:
        fac_body.append('c --- FAC colon2 -> sampson ionize (first 5 params)')
        fac_body.append('c --- note: managed/detail-buffer ions use FAC-based hybrid ionization rates here')

        # detail ions: direct remap
        for ln in fac_sections.get('colon2', []):
            s = ln.strip()
            if not s.startswith('d '):
                continue
            p = s.split()
            ion_from = int(p[1])
            if ion_from not in state['detail_ions']:
                continue
            lev_from_old = int(p[2])
            ion_to = int(p[3])
            lev_to_old = int(p[4])
            lev_from = _map_fac_level_for_model(state, ion_from, lev_from_old)
            if lev_from is None:
                continue
            lev_to = lev_to_old
            if ion_to in state['managed_ions']:
                lev_to = _map_fac_level_for_model(state, ion_to, lev_to_old)
                if lev_to is None:
                    continue
            # sampson ionize uses a 5-parameter format
            if len(p) < 10:
                continue
            params5 = p[5:10]
            fac_body.append(_format_rate_line(
                ion_from, lev_from, ion_to, lev_to, params5))

        # buffer ions: degeneracy-weighted averaging
        agg = {}
        for ln in fac_sections.get('colon2', []):
            s = ln.strip()
            if not s.startswith('d '):
                continue
            p = s.split()
            ion_from = int(p[1])
            if ion_from not in state['buffer_ions']:
                continue
            lev_from_old = int(p[2])
            ion_to = int(p[3])
            lev_to_old = int(p[4])
            lev_from = _map_fac_level_for_model(state, ion_from, lev_from_old)
            if lev_from is None:
                continue
            lev_to = lev_to_old
            if ion_to in state['managed_ions']:
                lev_to = _map_fac_level_for_model(state, ion_to, lev_to_old)
                if lev_to is None:
                    continue
            if len(p) < 10:
                continue
            params5 = [_safe_float(x) for x in p[5:10]]
            w = state['fac_deg'].get(ion_from, {}).get(lev_from_old, 1.0)
            _acc_weighted(agg, (ion_from, lev_from, ion_to, lev_to), params5, w)
        fac_body.extend(_emit_weighted_lines(agg))

    sh_lines = []
    if 'sampson ionize' in sh_sections:
        existing = _collect_transition_keys(fac_body)
        sh_lines.append('c --- SH supplemental sampson ionize (managed+SH remap, FAC precedence)')
        sh_lines.extend(_build_sh_supplemental_rates(
            sh_sections.get('sampson ionize', []), state,
            existing_keys=existing, positive_check=False))

    body = []
    if sh_lines:
        body.extend(sh_lines)
        if fac_body:
            body.append('')
    body.extend(fac_body)

    return _sec('sampson ionize', body)


def _collect_model_abs_energy(model_lines):
    """Build a map of absolute energy = enot(ion) + elev_energy from the model section."""
    enot = {}
    abs_e = {}
    cur = None
    for ln in model_lines:
        s = ln.strip()
        if s.startswith('enot'):
            p = s.split()
            if len(p) >= 4:
                cur = int(p[1])
                enot[cur] = _safe_float(p[3], 0.0)
            continue
        if s.startswith('elev') and cur is not None:
            p = s.split()
            if len(p) >= 6:
                lev = int(p[2])
                abs_e[(cur, lev)] = enot.get(cur, 0.0) + _safe_float(p[5], 0.0)
    return abs_e


def _collect_model_exc_energy(model_lines):
    """Build an excitation-energy map from the model section: (ion, lev) -> E_exc."""
    exc = {}
    cur = None
    for ln in model_lines:
        s = ln.strip()
        if s.startswith('enot'):
            p = s.split()
            if len(p) >= 2:
                cur = _safe_int(p[1], None)
            continue
        if s.startswith('elev') and cur is not None:
            p = s.split()
            if len(p) >= 6:
                lev = _safe_int(p[2], None)
                if lev is not None:
                    exc[(cur, lev)] = _safe_float(p[5], 0.0)
    return exc


def _count_decimal_places(num_str):
    s = str(num_str).strip().lower()
    if not s:
        return 0
    if 'e' in s:
        s = s.split('e', 1)[0]
    if '.' not in s:
        return 0
    return len(s.split('.', 1)[1])


def split_near_degenerate_model_energies(out_lines, tol=1e-9, delta=1e-3, digits=3):
    """
    Split identical or near-degenerate excitation energies within the model section.
    - Applied per ion
    - Split on the display-precision grid given by ``digits`` (default: 3)
    - Quantize the split interval to at least one display unit
    - Deterministic ordering: increasing lev
    """
    in_model = False
    cur_ion = None
    by_ion = defaultdict(list)  # ion -> [{'idx','lev','e','tokens'}]

    for idx, ln in enumerate(out_lines):
        s = ln.strip()
        if s.startswith('data '):
            in_model = (s[5:].strip() == 'model')
            continue
        if not in_model:
            continue
        if s == 'end data':
            break
        if s.startswith('enot'):
            p = s.split()
            if len(p) >= 2:
                cur_ion = _safe_int(p[1], None)
            continue
        if s.startswith('elev') and cur_ion is not None:
            p = s.split()
            if len(p) < 6:
                continue
            by_ion[cur_ion].append({
                'idx': idx,
                'lev': _safe_int(p[2], -1),
                'e': _safe_float(p[5], 0.0),
                'tokens': p,
            })

    touched_groups = 0
    touched_levels = 0
    for ion_id, recs in by_ion.items():
        if ion_id is None or ion_id <= 0 or len(recs) < 2:
            continue

        rs = sorted(recs, key=lambda r: (r['e'], r['lev']))
        groups = []
        cur = [rs[0]]
        for r in rs[1:]:
            if abs(r['e'] - cur[-1]['e']) <= tol:
                cur.append(r)
            else:
                groups.append(cur)
                cur = [r]
        groups.append(cur)

        for g in groups:
            if len(g) < 2:
                continue
            touched_groups += 1
            touched_levels += len(g)
            g_sorted = sorted(g, key=lambda r: r['lev'])
            eavg = sum(r['e'] for r in g_sorted) / len(g_sorted)
            m = len(g_sorted)

            if digits is None:
                dlist = [_count_decimal_places(r['tokens'][5]) for r in g_sorted]
                dlist.sort()
                g_digits = dlist[len(dlist) // 2] if dlist else 3
            else:
                g_digits = max(0, int(digits))
            unit = (10.0 ** (-g_digits)) if g_digits > 0 else 1.0

            # Even step-units ensure that, for even m, half-steps also lie on the display grid
            step_units = max(1, int(math.ceil(max(delta, unit) / unit)))
            if step_units % 2 == 1:
                step_units += 1
            step = step_units * unit
            e_center = round(eavg / unit) * unit

            for k, r in enumerate(g_sorted):
                offset = (k - (m - 1) / 2.0) * step
                new_e = e_center + offset
                p2 = list(r['tokens'])
                p2[5] = f'{new_e:.{g_digits}f}'
                occ22 = [_safe_int(x, 0) for x in p2[6:-1]]
                if len(occ22) < _MODEL_OCC_COLS:
                    occ22 += [0] * (_MODEL_OCC_COLS - len(occ22))
                else:
                    occ22 = occ22[:_MODEL_OCC_COLS]
                out_lines[r['idx']] = _format_elev(p2, occ22, p2[-1])

    return out_lines, {'groups': touched_groups, 'levels': touched_levels}


def filter_sampson_ionize_nonpositive_exc(out_lines, eps=1e-12):
    """
    Remove sampson ionize d-lines whose model excitation energies satisfy
    E_exc(to) <= E_exc(from).
    """
    sections = _parse_sections_from_lines(out_lines)
    model_lines = sections.get('model', [])
    if not model_lines or 'sampson ionize' not in sections:
        return out_lines, 0

    exc = _collect_model_exc_energy(model_lines)
    removed = 0
    out2 = []
    cur = None
    for ln in out_lines:
        s = ln.strip()
        if s.startswith('data '):
            cur = s[5:].strip()
            out2.append(ln)
            continue

        if cur == 'sampson ionize' and s.startswith('d '):
            p = s.split()
            if len(p) >= 5:
                ion_from = _safe_int(p[1], -1)
                lev_from = _safe_int(p[2], -1)
                ion_to = _safe_int(p[3], -1)
                lev_to = _safe_int(p[4], -1)
                ef = exc.get((ion_from, lev_from))
                et = exc.get((ion_to, lev_to))
                if ef is not None and et is not None and et <= ef + eps:
                    removed += 1
                    continue

        out2.append(ln)
    return out2, removed


def filter_nonpositive_transition_energy(out_lines, sections_to_filter=None, eps=1e-9):
    """
    Remove transitions in selected sections whose absolute energies satisfy
    abs energy(to) <= abs energy(from).
    This is used to prevent Cretin zero/negative transition-energy errors.
    """
    targets = set(sections_to_filter or ['phis', 'sampson excite', 'sampson ionize'])
    secs = _parse_sections_from_lines(out_lines)
    model_lines = secs.get('model', [])
    abs_e = _collect_model_abs_energy(model_lines)

    removed = defaultdict(int)
    out2 = []
    cur = None
    for ln in out_lines:
        s = ln.strip()
        if s.startswith('data '):
            cur = s[5:].strip()
            out2.append(ln)
            continue

        if cur in targets and s.startswith('d '):
            p = s.split()
            if len(p) >= 5:
                ion_from = _safe_int(p[1], -1)
                lev_from = _safe_int(p[2], -1)
                ion_to = _safe_int(p[3], -1)
                lev_to = _safe_int(p[4], -1)
                ef = abs_e.get((ion_from, lev_from))
                et = abs_e.get((ion_to, lev_to))
                if ef is not None and et is not None and et <= ef + eps:
                    removed[cur] += 1
                    continue

        out2.append(ln)
    return out2, dict(removed)


def repair_sampson_ionize_targets(out_lines, state, eps=1e-9):
    """
    Repair sampson ionize transitions that produce zero/negative thresholds.
    - Only target hybrid/remapped transitions involving managed ions
    - If ion_to is a managed ion, reassign to another target level that gives
      a positive threshold
    - If no such level exists, drop the transition
    """
    if not state:
        return out_lines, {'remapped': 0, 'dropped': 0}

    secs = _parse_sections_from_lines(out_lines)
    model_lines = secs.get('model', [])
    if not model_lines or 'sampson ionize' not in secs:
        return out_lines, {'remapped': 0, 'dropped': 0}

    abs_e = _collect_model_abs_energy(model_lines)
    managed_ions = set(state.get('managed_ions', set()))
    remapped = 0
    dropped = 0
    out2 = []
    cur = None
    seen = set()

    for ln in out_lines:
        s = ln.strip()
        if s.startswith('data '):
            cur = s[5:].strip()
            out2.append(ln)
            continue
        if s.startswith('end data'):
            cur = None
            out2.append(ln)
            continue
        if cur != 'sampson ionize' or not s.startswith('d '):
            out2.append(ln)
            continue

        p = s.split()
        if len(p) < 5:
            out2.append(ln)
            continue

        ion_from = _safe_int(p[1], -1)
        lev_from = _safe_int(p[2], -1)
        ion_to = _safe_int(p[3], -1)
        lev_to = _safe_int(p[4], -1)

        # Only repair hybrid/remapped transitions that involve managed
        # (detail/buffer) ions. Leave original SH-only transitions untouched.
        if ion_from not in managed_ions and ion_to not in managed_ions:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                out2.append('  ' + '  '.join(p))
            continue

        ef = abs_e.get((ion_from, lev_from))
        et = abs_e.get((ion_to, lev_to))
        new_lev_to = lev_to

        if ef is not None and et is not None and et <= ef + eps:
            candidates = []
            for cand in sorted(state['model_valid_levels'].get(ion_to, set())):
                e_cand = abs_e.get((ion_to, cand))
                if e_cand is None or e_cand <= ef + eps:
                    continue
                candidates.append((e_cand - ef, abs(cand - lev_to), cand))
            if candidates:
                candidates.sort()
                new_lev_to = candidates[0][2]
                p[4] = str(new_lev_to)
                remapped += 1
            else:
                dropped += 1
                continue

        key = tuple(p)
        if key in seen:
            continue
        seen.add(key)
        out2.append('  ' + '  '.join(p))

    return out2, {'remapped': remapped, 'dropped': dropped}


def sort_transition_lines_by_ion(out_lines, sections_to_sort=None):
    """
    Reorder d-lines inside each section by increasing ion_from.
    Default sort key: (ion_from, ion_to, lev_from, lev_to, raw text)
    """
    targets = set(sections_to_sort or [
        'phxs', 'phis', 'phot_ion',
        'colex2', 'colon2',
        'sampson excite', 'sampson ionize',
        'augxs',
    ])

    out = []
    i = 0
    n = len(out_lines)
    while i < n:
        ln = out_lines[i]
        s = ln.strip()
        if not s.startswith('data '):
            out.append(ln)
            i += 1
            continue

        sec_name = s[5:].strip()
        data_line = ln
        i += 1
        body = []
        while i < n and out_lines[i].strip() != 'end data':
            body.append(out_lines[i])
            i += 1
        end_line = out_lines[i] if i < n else 'end data'
        if i < n:
            i += 1

        if sec_name in targets:
            non_d = []
            d_rows = []
            for row in body:
                rs = row.strip()
                if not rs.startswith('d '):
                    non_d.append(row)
                    continue
                p = rs.split()
                ion_from = _safe_int(p[1], 10**9) if len(p) > 1 else 10**9
                lev_from = _safe_int(p[2], 10**9) if len(p) > 2 else 10**9
                ion_to = _safe_int(p[3], 10**9) if len(p) > 3 else 10**9
                lev_to = _safe_int(p[4], 10**9) if len(p) > 4 else 10**9
                d_rows.append((ion_from, ion_to, lev_from, lev_to, rs, row))
            d_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
            body = non_d + [r[-1] for r in d_rows]

        out.append(data_line)
        out.extend(body)
        out.append(end_line)

    return out


def merge_transition_sections(out_lines, dst_name, src_name, comment=None):
    """
    Move (merge) d-lines from the src section into the dst section and remove src.
    """
    blocks = []
    i = 0
    n = len(out_lines)
    while i < n:
        ln = out_lines[i]
        s = ln.strip()
        if not s.startswith('data '):
            blocks.append({'type': 'raw', 'line': ln})
            i += 1
            continue
        sec_name = s[5:].strip()
        sec_lines = [ln]
        i += 1
        while i < n:
            sec_lines.append(out_lines[i])
            if out_lines[i].strip() == 'end data':
                i += 1
                break
            i += 1
        blocks.append({'type': 'sec', 'name': sec_name, 'lines': sec_lines})

    dst_idx = None
    src_idx = None
    for idx, b in enumerate(blocks):
        if b['type'] != 'sec':
            continue
        if b['name'] == dst_name and dst_idx is None:
            dst_idx = idx
        if b['name'] == src_name and src_idx is None:
            src_idx = idx

    if dst_idx is None or src_idx is None:
        return out_lines, 0

    src_lines = blocks[src_idx]['lines']
    src_d = [ln for ln in src_lines if ln.strip().startswith('d ')]
    if not src_d:
        # Remove only the src section
        blocks.pop(src_idx)
        new_out = []
        for b in blocks:
            if b['type'] == 'raw':
                new_out.append(b['line'])
            else:
                new_out.extend(b['lines'])
        return new_out, 0

    # Account for dst index changes after removing src
    if src_idx < dst_idx:
        dst_idx -= 1
    blocks.pop(src_idx)

    dst_lines = blocks[dst_idx]['lines']
    end_pos = None
    for k, ln in enumerate(dst_lines):
        if ln.strip() == 'end data':
            end_pos = k
            break
    if end_pos is None:
        return out_lines, 0

    ins = []
    if comment:
        ins.append(comment)
    ins.extend(src_d)
    if ins and (end_pos > 0 and dst_lines[end_pos - 1].strip() != ''):
        ins.insert(0, '')
    dst_lines = dst_lines[:end_pos] + ins + dst_lines[end_pos:]
    blocks[dst_idx]['lines'] = dst_lines

    new_out = []
    for b in blocks:
        if b['type'] == 'raw':
            new_out.append(b['line'])
        else:
            new_out.extend(b['lines'])
    return new_out, len(src_d)


def drop_overlapping_transition_pairs(out_lines, keep_section, drop_section):
    """
    Remove d-lines from drop_section when they share the same (ion_from, ion_to)
    ion pair as keep_section.
    This avoids semantic overlap between sections at the ion-pair level.
    """
    keep_pairs = set()
    cur = None
    for ln in out_lines:
        s = ln.strip()
        if s.startswith('data '):
            cur = s[5:].strip()
            continue
        if s.startswith('end data'):
            cur = None
            continue
        if cur != keep_section or not s.startswith('d '):
            continue
        p = s.split()
        if len(p) >= 4:
            keep_pairs.add((_safe_int(p[1], -1), _safe_int(p[3], -1)))

    if not keep_pairs:
        return out_lines, 0

    out2 = []
    cur = None
    removed = 0
    for ln in out_lines:
        s = ln.strip()
        if s.startswith('data '):
            cur = s[5:].strip()
            out2.append(ln)
            continue
        if s.startswith('end data'):
            cur = None
            out2.append(ln)
            continue
        if cur == drop_section and s.startswith('d '):
            p = s.split()
            if len(p) >= 4:
                pair = (_safe_int(p[1], -1), _safe_int(p[3], -1))
                if pair in keep_pairs:
                    removed += 1
                    continue
        out2.append(ln)
    return out2, removed


def drop_empty_sections(out_lines, section_names):
    """
    Remove output sections that contain no d-lines.
    This cleans up placeholder sections that only contain comments or blank lines.
    """
    targets = set(section_names or [])
    out = []
    i = 0
    n = len(out_lines)
    removed = []
    while i < n:
        ln = out_lines[i]
        s = ln.strip()
        if not s.startswith('data '):
            out.append(ln)
            i += 1
            continue

        sec_name = s[5:].strip()
        block = [ln]
        i += 1
        has_d = False
        while i < n:
            block.append(out_lines[i])
            bs = out_lines[i].strip()
            if bs.startswith('d '):
                has_d = True
            if bs == 'end data':
                i += 1
                break
            i += 1

        if sec_name in targets and not has_d:
            removed.append(sec_name)
            continue
        out.extend(block)
    return out, removed


def _parse_sections_from_lines(lines):
    sections = {}
    cur = None
    for ln in lines:
        s = ln.strip()
        if s.startswith('data '):
            cur = s[5:].strip()
            sections[cur] = [ln]
            continue
        if cur is not None:
            sections[cur].append(ln)
            if s == 'end data':
                cur = None
    return sections


def collect_model_valid_levels(model_lines):
    valid = defaultdict(set)
    cur = None
    for ln in model_lines:
        s = ln.strip()
        if s.startswith('enot'):
            p = s.split()
            if len(p) >= 2:
                cur = int(p[1])
                valid[cur]
            continue
        if s.startswith('elev') and cur is not None:
            p = s.split()
            if len(p) >= 3:
                valid[cur].add(int(p[2]))
    return valid


def validate_section_level_refs(section_name, section_lines, valid_levels):
    errs = []
    seen = set()
    for idx, ln in enumerate(section_lines, start=1):
        s = ln.strip()
        if not s.startswith('d '):
            continue
        p = s.split()
        if len(p) < 5:
            errs.append(f'{section_name}:{idx} malformed d-line')
            continue
        try:
            ion_from = int(p[1]); lev_from = int(p[2])
            ion_to = int(p[3]); lev_to = int(p[4])
        except ValueError:
            errs.append(f'{section_name}:{idx} non-integer ion/lev')
            continue

        if lev_from not in valid_levels.get(ion_from, set()):
            errs.append(f'{section_name}:{idx} invalid from ({ion_from},{lev_from})')
        if lev_to not in valid_levels.get(ion_to, set()):
            errs.append(f'{section_name}:{idx} invalid to ({ion_to},{lev_to})')

        key = (ion_from, lev_from, ion_to, lev_to, tuple(p[5:]))
        if key in seen:
            errs.append(f'{section_name}:{idx} duplicate d-line')
        seen.add(key)
    return errs


def validate_output_sections(out_lines):
    sections = _parse_sections_from_lines(out_lines)
    model = sections.get('model', [])
    if not model:
        return ['missing model section']

    valid = collect_model_valid_levels(model)
    targets = ['phxs', 'phis', 'phot_ion', 'colex2', 'colon2',
               'augxs', 'sampson ionize', 'sampson excite']
    errs = []
    for name in targets:
        if name in sections:
            errs.extend(validate_section_level_refs(name, sections[name], valid))
    return errs


def validate_no_managed_sampson_ionize_sources(out_lines, managed_ions):
    """Disallow Phase-2 lines that use a managed ion as a sampson ionize source."""
    if not managed_ions:
        return []
    bad = []
    cur = None
    for idx, ln in enumerate(out_lines, start=1):
        s = ln.strip()
        if s.startswith('data '):
            cur = s[5:].strip()
            continue
        if cur == 'sampson ionize' and s.startswith('d '):
            p = s.split()
            if len(p) >= 2:
                ion_from = _safe_int(p[1], -1)
                if ion_from in managed_ions:
                    bad.append(f'sampson ionize:{idx} managed ion_from={ion_from}')
    return bad


def _section_has_d_lines(sections, name):
    for ln in sections.get(name, []):
        if ln.strip().startswith('d '):
            return True
    return False


def _normalize_photo_section_alias(sections):
    """
    Normalize photoionization section aliases:
      - if phis is absent and only phot_ion exists, create a phis alias
    """
    out = dict(sections)
    if 'phis' not in out and 'phot_ion' in out:
        out['phis'] = out['phot_ion']
    return out


def _pick_photo_output_name(sh_sections, fac_sections):
    """Choose the output photoionization section name, preferring the input style."""
    if 'phis' in sh_sections:
        return 'phis'
    if 'phot_ion' in sh_sections:
        return 'phot_ion'
    if 'phis' in fac_sections:
        return 'phis'
    if 'phot_ion' in fac_sections:
        return 'phot_ion'
    return 'phis'


def _rename_data_section(lines, new_name):
    """Rename only the data-section header in the result returned by _sec()."""
    if not lines:
        return lines
    out = []
    renamed = False
    for ln in lines:
        s = ln.strip()
        if not renamed and s.startswith('data '):
            out.append(f'data    {new_name}')
            renamed = True
        else:
            out.append(ln)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def merge(sh_path, fac_path, phase, detail_ions, buffer_ions, n_super, output_path,
          split_tol=1e-9, split_delta=1e-3, split_digits=3,
          sampson_source='fac-colon2',
          filter_nonpositive=False,
          photoionization='auto'):
    fac_ion_ids = set(detail_ions or []) | set(buffer_ions or [])
    print(f'[merge_cretin]  Phase={phase}  detail={sorted(detail_ions)}'
          f'  buffer={sorted(buffer_ions)}  n_super={n_super}')
    print(f'  SH : {sh_path}')
    print(f'  FAC: {fac_path}')
    print(f'  Out: {output_path}\n')

    sh  = parse_sections(sh_path)
    fac = parse_sections(fac_path)
    sh_photo  = _normalize_photo_section_alias(sh)
    fac_photo = _normalize_photo_section_alias(fac)
    photo_output_name = _pick_photo_output_name(sh, fac)
    print(f'  SH sections : {list(sh.keys())}')
    print(f'  FAC sections: {list(fac.keys())}\n')
    fac_has_phis = _section_has_d_lines(fac_photo, 'phis')
    sh_has_phis = _section_has_d_lines(sh_photo, 'phis')
    include_phis = ((photoionization == 'on') or
                    (photoionization == 'auto' and (fac_has_phis or sh_has_phis)))
    print(f'  photo mode={photoionization}  fac_has_d={fac_has_phis}  '
          f'sh_has_d={sh_has_phis}  include={include_phis}  '
          f'out_section={photo_output_name}\n')

    out = []
    out += [f'c  Z=29 (Cu)  FAC ions={sorted(fac_ion_ids)} + SH backbone',
            f'c  Phase {phase}  n_super={n_super}'
            f'  SH={os.path.basename(sh_path)}'
            f'  FAC={os.path.basename(fac_path)}', '']

    def add(lines, label):
        print(f'[{label}]')
        if lines:
            out.extend(lines)
            out.append('')

    phase2_state = None

    if phase == 1:
        add(sec_model_p1(sh),                              'model  SH as-is')
        add(sec_phxs_p1(sh),                               'phxs   SH as-is')
        if include_phis:
            photo_lines = sec_phis(sh_photo, fac_photo, fac_ion_ids)
            if photo_output_name != 'phis':
                photo_lines = _rename_data_section(photo_lines, photo_output_name)
            add(photo_lines, f'{photo_output_name}   FAC/SH mixed')
        else:
            print(f'[{photo_output_name}] skipped')
    else:
        model_lines, phase2_state = sec_model_p2_hybrid(
            sh, fac, detail_ions, buffer_ions, n_super)
        managed = sorted(phase2_state['managed_ions'])
        add(model_lines, f'model  detail/buffer/SH hybrid  managed={managed}')

        add(sec_phxs_p2_hybrid(sh, fac, phase2_state),
            'phxs   FAC(detail+buffer avg) + SH')
        if include_phis:
            photo_lines = sec_phis_p2_hybrid(sh_photo, fac_photo, phase2_state)
            if photo_output_name != 'phis':
                photo_lines = _rename_data_section(photo_lines, photo_output_name)
            add(photo_lines, f'{photo_output_name}   FAC(detail+buffer avg) + SH remap/distribution')
        else:
            print(f'[{photo_output_name}] skipped')
        add(_sec_collisional_p2_hybrid('colex2', fac, phase2_state),
            'colex2 FAC(detail+buffer avg)')
        add(_sec_collisional_p2_hybrid('colon2', fac, phase2_state),
            'colon2 FAC(detail+buffer avg)')

    if phase == 1:
        add(sec_colex2(fac, fac_ion_ids),                  'colex2 NEW FAC CE')
        add(sec_colon2(fac, fac_ion_ids),                  'colon2 NEW FAC CI')
        add(sec_sampson_excite(sh),                        'sampson excite  SH as-is')
        add(sec_sampson_ionize_p1(sh),                     'sampson ionize  SH as-is')
    else:
        add(sec_sampson_excite_p2_hybrid(sh, phase2_state),
            'sampson excite  SH remap + FAC coexist')
        if sampson_source == 'fac-colon2':
            add(sec_sampson_ionize_from_fac_colon2_p2(sh, fac, phase2_state),
                'sampson ionize  FAC colon2 + SH remap supplement')
        else:
            add(sec_sampson_ionize_p2_hybrid(sh, phase2_state),
                'sampson ionize  SH boundary -> managed remap')

    if phase == 1:
        add(sec_augxs(sh, fac, fac_ion_ids), 'augxs  FAC+SH mixed')
    else:
        add(sec_augxs_p2_hybrid(sh, fac, phase2_state),
            'augxs  FAC(detail+buffer avg) + SH')
    add(sec_augis(sh),                                     'augis  SH as-is')

    if phase == 2:
        out, split_stats = split_near_degenerate_model_energies(
            out, tol=split_tol, delta=split_delta, digits=split_digits)
        if split_stats['groups'] > 0:
            print('  [perturb] split near-degenerate model levels -> '
                  f"groups={split_stats['groups']} levels={split_stats['levels']} "
                  f'(tol={split_tol:g}, delta={split_delta:g} eV, digits={split_digits})')

        # NOTE:
        # If sampson ionize is filtered using excitation-energy non-positivity,
        # physically valid ionization channels (for example 1->0) may also be removed.
        # The default pipeline therefore does not apply that filter.

        if filter_nonpositive:
            out, removed = filter_nonpositive_transition_energy(
                out, sections_to_filter=['phxs', photo_output_name, 'colex2',
                                         'sampson excite', 'sampson ionize'])
            if removed:
                rm_msg = ', '.join(f'{k}:{v}' for k, v in sorted(removed.items()))
                print(f'  [filter] removed non-positive transitions -> {rm_msg}')

        validation_errors = validate_output_sections(out)
        if sampson_source == 'sh-boundary':
            validation_errors.extend(
                validate_no_managed_sampson_ionize_sources(
                    out, phase2_state.get('managed_ions', set())))
        if validation_errors:
            print('\n[validate] level-reference errors detected:', file=sys.stderr)
            for msg in validation_errors[:30]:
                print(f'  - {msg}', file=sys.stderr)
            if len(validation_errors) > 30:
                print(f'  ... {len(validation_errors) - 30} more', file=sys.stderr)
            raise RuntimeError('output failed section/reference validation')

    out, removed_overlap = drop_overlapping_transition_pairs(
        out, keep_section='sampson ionize', drop_section='colon2')
    if removed_overlap > 0:
        print(f'  [dedup] removed colon2 pairs covered by sampson ionize -> {removed_overlap}')

    out, sampson_fix = repair_sampson_ionize_targets(out, phase2_state)
    if sampson_fix['remapped'] or sampson_fix['dropped']:
        print('  [sampson-fix] repaired sampson ionize targets -> '
              f"remapped={sampson_fix['remapped']} dropped={sampson_fix['dropped']}")

    out, removed_empty = drop_empty_sections(out, ['colon2'])
    if removed_empty:
        print(f'  [cleanup] removed empty sections -> {", ".join(removed_empty)}')

    out = sort_transition_lines_by_ion(out)
    print('  [order] sorted d-lines by ion_from (h-like first)')

    with open(output_path, 'w') as fh:
        fh.write('\n'.join(out) + '\n')

    print(f'\n[✓] {output_path}  ({len(out)} lines)')
    print('\nD-line counts by section:')
    cur = None
    counts = {}
    for ln in out:
        s = ln.strip()
        if s.startswith('data '):
            cur = s[5:]
            counts[cur] = 0
        elif cur and s.startswith('d '):
            counts[cur] += 1
    for k, v in counts.items():
        print(f'  {k:25s}: {v:6d}')


def _infer_phase2_default_ions(fac_path):
    """
    Default automatic ion selection for Phase 2:
      - minimum/maximum FAC model ion IDs -> buffer
      - ions in between -> detail
    """
    fallback_detail, fallback_buffer = {1, 2}, {3}
    try:
        fac_sections = parse_sections(fac_path)
        fac_ions = sorted(parse_model_ions(fac_sections.get('model', [])).keys())
    except Exception as exc:
        print(f'  [warn] failed to parse FAC model for auto ions: {exc}',
              file=sys.stderr)
        return fallback_detail, fallback_buffer

    if not fac_ions:
        print('  [warn] FAC model has no ions; fallback to detail={1,2} buffer={3}',
              file=sys.stderr)
        return fallback_detail, fallback_buffer
    if len(fac_ions) == 1:
        return {fac_ions[0]}, set()
    if len(fac_ions) == 2:
        return set(), set(fac_ions)
    return set(fac_ions[1:-1]), {fac_ions[0], fac_ions[-1]}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sh', default='dca_29',
                    help='Screened-Hydrogenic input file (default: dca_29)')
    ap.add_argument('--fac', default='Cu_atom_5charge.dat',
                    help='FAC input file (default: Cu_atom_5charge.dat)')
    ap.add_argument('--phase', type=int, default=1, choices=[1, 2],
                    help='1=keep SH backbone, 2=replace with FAC levels (default: 1)')
    ap.add_argument('--ion-min', type=int, default=1,
                    help='Minimum ion_id to treat with FAC (Phase 1 default: 1)')
    ap.add_argument('--ion-max', type=int, default=3,
                    help='Maximum ion_id to treat with FAC (Phase 1 default: 3)')
    ap.add_argument('--detail-ions', nargs='*', type=int, default=None,
                    help='Phase 2 detail-basis FAC ions (e.g. 17 18 19)')
    ap.add_argument('--buffer-ions', nargs='*', type=int, default=None,
                    help='Phase 2 buffer-basis FAC ions (same-n averaged, e.g. 16 20)')
    ap.add_argument('--n-super', type=int, default=4,
                    help='Use SH superconfiguration for levels with n >= N (default: 4)')
    ap.add_argument('--split-degen-tol', type=float, default=1e-9,
                    help='Phase 2: tolerance for same/near-degenerate model energies (eV)')
    ap.add_argument('--split-degen-delta', type=float, default=1e-3,
                    help='Phase 2: degeneracy-splitting interval delta (eV)')
    ap.add_argument('--split-degen-digits', type=int, default=3,
                    help='Phase 2: decimal digits for split energy output (default: 3)')
    ap.add_argument('--sampson-source', choices=['fac-colon2', 'sh-boundary'],
                    default='fac-colon2',
                    help='Phase 2 sampson ionize source (default: fac-colon2)')
    ap.add_argument('--photoionization', choices=['auto', 'on', 'off'],
                    default='auto',
                    help=('phis/phot_ion handling: auto=include when FAC or SH has d-lines (default), '
                          'on=always include, off=always exclude'))
    ap.add_argument('--filter-nonpositive', action='store_true',
                    help='Phase 2: remove non-positive transition lines (default: off)')
    ap.add_argument('-o', '--output', default='Cu_cretin_merged.dat')
    args = ap.parse_args()

    if args.phase == 2:
        if args.detail_ions is None and args.buffer_ions is None:
            detail_ions, buffer_ions = _infer_phase2_default_ions(args.fac)
            print(f'[auto] phase2 ions from FAC model: detail={sorted(detail_ions)} '
                  f'buffer={sorted(buffer_ions)}')
        else:
            detail_ions = set(args.detail_ions or [])
            buffer_ions = set(args.buffer_ions or [])
    else:
        fac_ion_ids = set(range(args.ion_min, args.ion_max + 1))
        if args.detail_ions is not None or args.buffer_ions is not None:
            fac_ion_ids = set(args.detail_ions or []) | set(args.buffer_ions or [])
        detail_ions = fac_ion_ids
        buffer_ions = set()

    merge(args.sh, args.fac, args.phase, detail_ions, buffer_ions,
          args.n_super, args.output,
          split_tol=args.split_degen_tol,
          split_delta=args.split_degen_delta,
          split_digits=(None if args.split_degen_digits < 0 else args.split_degen_digits),
          sampson_source=args.sampson_source,
          filter_nonpositive=args.filter_nonpositive,
          photoionization=args.photoionization)


if __name__ == '__main__':
    main()
