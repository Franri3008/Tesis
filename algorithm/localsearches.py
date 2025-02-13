import random

def MejorarAfinidad_primario(sol, OT, AOR, nSlot, nDays, surgeon, second, SP, dictCosts, fichas, hablar=False):
    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy()
    scheduled = [p for p in range(len(pacientes)) if pacientes[p] != -1]
    if not scheduled:
        return (pacientes, primarios, secundarios), "MejorarAfinidad_primario"

    feasible_changes = []
    for p in scheduled:
        start_blk = pacientes[p]
        dur = OT[p]
        main_current = primarios[start_blk]
        sec_current = secundarios[start_blk]
        current_cost = dictCosts[(main_current, sec_current, start_blk)]

        best_s = None
        best_val = current_cost
        o_ = start_blk // (nSlot * nDays)
        tmp = start_blk % (nSlot * nDays)
        d_ = tmp // nSlot

        # Look for a better main surgeon
        for s_cand in surgeon:
            if SP[p][s_cand] == 1 and s_cand != sec_current:
                cost_cand = dictCosts[(s_cand, sec_current, start_blk)]
                if cost_cand > best_val:
                    if fichas[(s_cand, d_)] >= cost_cand:
                        if cost_cand > best_val:
                            best_val = cost_cand
                            best_s = s_cand

        if best_s is not None:
            feasible_changes.append((p, best_s))

    if not feasible_changes:
        return (pacientes, primarios, secundarios), "MejorarAfinidad_primario"

    p_sel, s_sel = random.choice(feasible_changes)
    start_blk = pacientes[p_sel]
    dur = OT[p_sel]
    if hablar:
        old_s = primarios[start_blk]
        sec_s = secundarios[start_blk]
        print(f"[MejorarAfinidad_primario] p={p_sel} old_main={old_s} new_main={s_sel} sec={sec_s}")

    o_ = start_blk // (nSlot * nDays)
    tmp = start_blk % (nSlot * nDays)
    d_ = tmp // nSlot
    cost_new = dictCosts[(s_sel, secundarios[start_blk], start_blk)]
    fichas[(s_sel, d_)] -= cost_new

    for b in range(dur):
        primarios[start_blk + b] = s_sel

    return (pacientes, primarios, secundarios), "MejorarAfinidad_primario"

def MejorarAfinidad_secundario(sol, OT, AOR, nSlot, nDays, surgeon, second, SP, dictCosts, fichas, hablar=False):
    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy()
    scheduled = [p for p in range(len(pacientes)) if pacientes[p] != -1]
    if not scheduled:
        return (pacientes, primarios, secundarios), "MejorarAfinidad_secundario"

    feasible_changes = []
    for p in scheduled:
        start_blk = pacientes[p]
        dur = OT[p]
        main_current = primarios[start_blk]
        sec_current = secundarios[start_blk]
        current_cost = dictCosts[(main_current, sec_current, start_blk)]

        best_s = None
        best_val = current_cost
        o_ = start_blk // (nSlot * nDays)
        tmp = start_blk % (nSlot * nDays)
        d_ = tmp // nSlot

        for s2 in second:
            if s2 != main_current:
                cand_val = dictCosts[(main_current, s2, start_blk)]
                if cand_val > best_val:
                    # typically no fichas check for secondary, so we skip that
                    # but if we do, we can add it. We'll assume only main surgeons use fichas.
                    if cand_val > best_val:
                        best_val = cand_val
                        best_s = s2

        if best_s is not None:
            feasible_changes.append((p, best_s))

    if not feasible_changes:
        return (pacientes, primarios, secundarios), "MejorarAfinidad_secundario"

    p_sel, s_sel = random.choice(feasible_changes)
    start_blk = pacientes[p_sel]
    dur = OT[p_sel]
    if hablar:
        old_s = secundarios[start_blk]
        main_s = primarios[start_blk]
        print(f"[MejorarAfinidad_secundario] p={p_sel} main={main_s} old_sec={old_s} new_sec={s_sel}")

    for b in range(dur):
        secundarios[start_blk + b] = s_sel

    return (pacientes, primarios, secundarios), "MejorarAfinidad_secundario"

def AdelantarDia(sol, OT, AOR, nSlot, nDays, surgeon, second, SP, dictCosts, fichas, hablar=False):
    def compressf(o, d, t):
        return o*nSlot*nDays + d*nSlot + t
    def decompressf(val):
        o_ = val // (nSlot*nDays)
        tmp = val % (nSlot*nDays)
        d_ = tmp // nSlot
        t_ = tmp % nSlot
        return o_, d_, t_

    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy()
    scheduled = [p for p in range(len(pacientes)) if pacientes[p] != -1]
    if not scheduled:
        return (pacientes, primarios, secundarios), "AdelantarDia"

    feasible_moves = []
    for p in scheduled:
        start_blk = pacientes[p]
        o, d, t = decompressf(start_blk)
        dur = OT[p]
        for new_d in range(d):
            can_move = True
            for b in range(dur):
                if AOR[p][o][t+b][new_d % 5] != 1:
                    can_move = False
                    break
                new_blk = compressf(o, new_d, t + b)
                if new_blk in primarios or new_blk in secundarios:
                    can_move = False
                    break
            if can_move:
                feasible_moves.append((p, o, d, t, new_d))

    if not feasible_moves:
        return (pacientes, primarios, secundarios), "AdelantarDia"

    p_sel, o_old, d_old, t_old, d_new = random.choice(feasible_moves)
    dur = OT[p_sel]
    start_blk_old = pacientes[p_sel]
    for b in range(dur):
        ob = start_blk_old + b
        if ob in primarios: del primarios[ob]
        if ob in secundarios: del secundarios[ob]

    new_start = compressf(o_old, d_new, t_old)
    possible_mains = [s for s in surgeon if SP[p_sel][s] == 1]
    if not possible_mains:
        return (pacientes, primarios, secundarios), "AdelantarDia"
    main_s = None
    sec_s = None
    for cand_m in possible_mains:
        cost_ = dictCosts.get((cand_m, None, new_start), 0)
        if fichas[(cand_m, d_new)] >= cost_:
            main_s = cand_m
            fichas[(cand_m, d_new)] -= cost_
            break
    if main_s is None:
        for b in range(dur):
            primarios[start_blk_old + b] = primarios.get(start_blk_old, None)
            secundarios[start_blk_old + b] = secundarios.get(start_blk_old, None)
        return (pacientes, primarios, secundarios), "AdelantarDia"
    sec_s = random.choice(second)
    for b in range(dur):
        primarios[new_start + b] = main_s
        secundarios[new_start + b] = sec_s
    pacientes[p_sel] = new_start

    if hablar:
        print(f"[AdelantarDia] p={p_sel} from day {d_old} to {d_new}, OR={o_old}, slot={t_old}")

    return (pacientes, primarios, secundarios), "AdelantarDia"

def MejorOR(sol, OT, AOR, nSlot, nDays, room, surgeon, second, SP, dictCosts, fichas, hablar=False):
    def compressf(o, d, t):
        return o*nSlot*nDays + d*nSlot + t
    def decompressf(val):
        o_ = val // (nSlot*nDays)
        tmp = val % (nSlot*nDays)
        d_ = tmp // nSlot
        t_ = tmp % nSlot
        return o_, d_, t_

    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy()
    scheduled = [p for p in range(len(pacientes)) if pacientes[p] != -1]
    if not scheduled:
        return (pacientes, primarios, secundarios), "MejorOR"

    feasible_moves = []
    for p in scheduled:
        old_start = pacientes[p]
        o_old, d_old, t_old = decompressf(old_start)
        dur = OT[p]
        main_old = primarios[old_start]
        sec_old = secundarios[old_start]
        for o_new in room:
            if o_new == o_old:
                continue
            can_move = True
            for b in range(dur):
                if AOR[p][o_new][t_old + b][d_old % 5] != 1:
                    can_move = False
                    break
                nb = compressf(o_new, d_old, t_old + b)
                if nb in primarios or nb in secundarios:
                    can_move = False
                    break
            if can_move:
                feasible_moves.append((p, o_old, d_old, t_old, o_new))

    if not feasible_moves:
        return (pacientes, primarios, secundarios), "MejorOR"

    p_sel, o_old, d_old, t_old, o_new = random.choice(feasible_moves)
    dur = OT[p_sel]
    start_blk_old = pacientes[p_sel]
    for b in range(dur):
        ob = start_blk_old + b
        if ob in primarios: del primarios[ob]
        if ob in secundarios: del secundarios[ob]
    new_start = compressf(o_new, d_old, t_old)
    possible_mains = [s for s in surgeon if SP[p_sel][s] == 1]
    if not possible_mains:
        return (pacientes, primarios, secundarios), "MejorOR"
    main_s = None
    sec_s = None
    for cand_m in possible_mains:
        cost_ = dictCosts.get((cand_m, None, new_start), 0)
        if fichas[(cand_m, d_old)] >= cost_:
            main_s = cand_m
            fichas[(cand_m, d_old)] -= cost_
            break
    if main_s is None:
        for b in range(dur):
            primarios[start_blk_old + b] = primarios.get(start_blk_old, None)
            secundarios[start_blk_old + b] = secundarios.get(start_blk_old, None)
        return (pacientes, primarios, secundarios), "MejorOR"
    sec_s = random.choice(second)
    for b in range(dur):
        primarios[new_start + b] = main_s
        secundarios[new_start + b] = sec_s
    pacientes[p_sel] = new_start

    if hablar:
        print(f"[MejorOR] p={p_sel} from OR={o_old} to OR={o_new}, day={d_old}, slot={t_old}")

    return (pacientes, primarios, secundarios), "MejorOR"