import random
import copy

def compress(o, d, t, nSlot, nDays):
    return o * nSlot * nDays + d * nSlot + t

def decompress(val, nSlot, nDays):
    o = val // (nSlot * nDays)
    temp = val % (nSlot * nDays)
    d = temp // nSlot
    t = temp % nSlot
    return o, d, t

def is_feasible_block(p, o, d, t, AOR):
    if AOR[p][o][t][d % 5] == 1:
        return True
    return False

def MejorarAfinidad_primario(solucion, surgeon, second, OT, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    #sol = (solucion[0][0].copy(), solucion[0][1].copy(), solucion[0][2].copy());
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    #pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);
    scheduled = [p for p in range(len(pacientes_copy)) if pacientes_copy[p] != -1];
    if not scheduled:
        #print("[MejorarAfinidad_primario] No hay pacientes programados.") if hablar else None;
        return solucion
    
    feasible_changes = [];
    for p_sel in scheduled:
        dur = int(OT[p_sel]);
        o, d, t = decompress(pacientes_copy[p_sel], nSlot, nDays);
        s_sel = primarios_copy[pacientes_copy[p_sel]];
        a_sel = secundarios_copy[pacientes_copy[p_sel]];
        candidates = [s for s in surgeon if SP[p_sel][s] == 1 and s != s_sel];
        
        for s in candidates:
            # Solo si tiene disponibilidad completa:
            if not all(surgeon_schedule_copy[s][d][t + b] == -1 for b in range(dur)):
                continue;
            # Si no tiene fichas suficientes, no se considera
            if not all(fichas_copy[(s, d_aux)] - dictCosts[(s, a_sel, pacientes_copy[p_sel])] >= 0 for d_aux in range(d, nDays)):
                continue;
            # Si no hay mejora, no se considera
            mejora = dictCosts[(s, a_sel, pacientes_copy[p_sel])] - dictCosts[s_sel, a_sel, pacientes_copy[p_sel]];
            if mejora <= 0:
                continue;
            feasible_changes.append((p_sel, s, s_sel, mejora));
    if len(feasible_changes) == 0:
        #print("[MejorarAfinidad_primario] No hay cambios realizables.") if hablar else None;
        return solucion

    feasible_changes.sort(key=lambda x: x[-1], reverse=True);
    p_sel, s_sel, s_old, mejora = feasible_changes[0];
    start_blk = pacientes_copy[p_sel]
    dur = OT[p_sel]
    #if hablar:
    #    old_s = primarios_copy[start_blk]
    #    sec_s = secundarios_copy[start_blk]
        #print(f"[MejorarAfinidad_primario] p={p_sel} old_main={old_s} new_main={s_sel} sec={sec_s}, mejora = {mejora}")

    o, d, t = decompress(start_blk, nSlot, nDays);
    cost_new = dictCosts[(s_sel, secundarios_copy[start_blk], start_blk)];
    cost_old = dictCosts[(s_old, secundarios_copy[start_blk], start_blk)];
    for d_aux in range(d, nDays):
        fichas_copy[(s_sel, d_aux)] -= cost_new;
        fichas_copy[(s_old, d_aux)] += cost_old;

    for b in range(dur):
        primarios_copy[start_blk + b] = s_sel;
        surgeon_schedule_copy[s_sel][d][t + b] = p_sel;
        surgeon_schedule_copy[s_old][d][t + b] = -1;

    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def MejorarAfinidad_secundario(solucion, surgeon, second, OT, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    #sol = (solucion[0][0].copy(), solucion[0][1].copy(), solucion[0][2].copy());
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    #pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);
    scheduled = [p for p in range(len(pacientes_copy)) if pacientes_copy[p] != -1];
    if not scheduled:
        #print("[MejorarAfinidad_secundario] No hay pacientes programados.") if hablar else None;
        return solucion
    
    feasible_changes = [];
    for p_sel in scheduled:
        dur = int(OT[p_sel]);
        o, d, t = decompress(pacientes_copy[p_sel], nSlot, nDays);
        s_sel = primarios_copy[pacientes_copy[p_sel]];
        a_sel = secundarios_copy[pacientes_copy[p_sel]];
        candidates = [a for a in second if a != a_sel];
        for a in candidates:
            # Solo si tiene disponibilidad completa:
            if not all(surgeon_schedule_copy[a][d][t + b] == -1 for b in range(dur)):
                continue;
            # Si no tiene fichas suficientes, no se considera
            if not all(fichas_copy[(s_sel, d_aux)] - dictCosts[(s_sel, a, pacientes_copy[p_sel])] >= 0 for d_aux in range(d, nDays)):
                continue;
            # Si no hay mejora, no se considera
            mejora = dictCosts[(s_sel, a, pacientes_copy[p_sel])] - dictCosts[s_sel, a_sel, pacientes_copy[p_sel]];
            if mejora <= 0:
                continue;
            feasible_changes.append((p_sel, a, a_sel, mejora));
    if len(feasible_changes) == 0:
        #print("[MejorarAfinidad_secundario] No hay cambios realizables.") if hablar else None;
        return solucion

    feasible_changes.sort(key=lambda x: x[-1], reverse=True);
    p_sel, a_sel, a_old, mejora = feasible_changes[0];
    start_blk = pacientes_copy[p_sel];
    dur = OT[p_sel];
    #if hablar:
    #    old_a = secundarios_copy[start_blk];
    #    main_s = primarios_copy[start_blk];
        #print(f"[MejorarAfinidad_secundario] p={p_sel} old_second={old_a} new_second={a_sel} main={main_s}, mejora = {mejora}");

    o, d, t = decompress(start_blk, nSlot, nDays);
    main_s = primarios_copy[start_blk];
    cost_new = dictCosts[(main_s, a_sel, start_blk)];
    cost_old = dictCosts[(main_s, a_old, start_blk)];
    for d_aux in range(d, nDays):
        fichas_copy[(main_s, d_aux)] = fichas_copy[(main_s, d_aux)] + (cost_old) - (cost_new);

    for b in range(dur):
        secundarios_copy[start_blk + b] = a_sel;
        surgeon_schedule_copy[a_sel][d][t + b] = p_sel;
        surgeon_schedule_copy[a_old][d][t + b] = -1;

    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

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

    #print(f"[AdelantarDia] p={p_sel} from day {d_old} to {d_new}, OR={o_old}, slot={t_old}") if hablar else None

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
    
    #print(f"[MejorOR] p={p_sel} from OR={o_old} to OR={o_new}, day={d_old}, slot={t_old}") if hablar else None;

    return (pacientes, primarios, secundarios), "MejorOR"