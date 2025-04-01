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

def AdelantarDia(solucion, surgeon, second, OT, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);
    scheduled = [p for p in range(len(pacientes_copy)) if pacientes_copy[p] != -1];
    if not scheduled:
        print("[AdelantarDia] No hay pacientes programados.") if hablar else None;
        return solucion

    feasible_moves = [];
    for p in scheduled:
        start_blk = pacientes_copy[p];
        s = primarios_copy[start_blk];
        a = secundarios_copy[start_blk];
        o, d, t = decompress(start_blk, nSlot, nDays);
        if d == 0:
            continue;
        dur = OT[p];
        for new_d in range(d):
            for new_t in range(nSlot):
                for new_o in range(len(or_schedule_copy)):
                    new_blk = compress(new_o, new_d, new_t, nSlot, nDays);  
                    if AOR[p][new_o][new_t][new_d % 5] != 1 or new_t + dur >= nSlot or (new_t < nSlot//2 and new_t + dur >= nSlot//2):
                        continue;
                    can_move = True;
                    # Solo si cirujanos tienen disponibilidad completa:
                    if not all(surgeon_schedule_copy[s][new_d][new_t + b] == -1 for b in range(dur)):
                        continue;
                    if not all(surgeon_schedule_copy[a][new_d][new_t + b] == -1 for b in range(dur)):
                        continue;
                    # Si no tiene fichas suficientes, no se considera
                    if not all(fichas_copy[(s, d_aux)] - dictCosts[(s, a, new_blk)] >= 0 for d_aux in range(new_d, nDays)):
                        continue;
                    for b in range(dur):
                        new_blk = compress(new_o, new_d, new_t + b, nSlot, nDays);
                        if new_blk in primarios_copy or new_blk in secundarios_copy:
                            can_move = False;
                            break
                    if can_move:
                        feasible_moves.append((p, o, new_o, t, new_t, d, new_d));
    if not feasible_moves:
        print("[AdelantarDia] No hay cambios realizables.") if hablar else None;
        return solucion

    p_sel, o_old, o_new, t_old, t_new, d_old, d_new = random.choice(feasible_moves);
    dur = OT[p_sel];
    start_blk_old = pacientes_copy[p_sel];
    s = primarios_copy[start_blk_old];
    a = secundarios_copy[start_blk_old];
    for b in range(dur):
        ob = start_blk_old + b;
        if ob in primarios_copy:
            surgeon_schedule_copy[primarios_copy[ob]][d_old][t_old + b] = -1;
            del primarios_copy[ob];
        if ob in secundarios_copy:
            surgeon_schedule_copy[secundarios_copy[ob]][d_old][t_old + b] = -1;
            del secundarios_copy[ob];
        or_schedule_copy[o_old][d_old][t_old + b] = -1;

    new_start = compress(o_new, d_new, t_new, nSlot, nDays);
    for d_aux in range(d_new, nDays):
        fichas_copy[(s, d_aux)] -= dictCosts[(s, a, new_start)];
    for b in range(dur):
        primarios_copy[new_start + b] = s;
        secundarios_copy[new_start + b] = a;
        surgeon_schedule_copy[s][d_new][t_new + b] = p_sel;
        surgeon_schedule_copy[a][d_new][t_new + b] = p_sel;
        or_schedule_copy[o_new][d_new][t_new + b] = p_sel;
    pacientes_copy[p_sel] = new_start;

    if hablar:
        print(f"[AdelantarDia] Patient {p_sel} moved from day {d_old} to day {d_new}, from OR {o_old} to OR {o_new}, and from slot {t_old} to slot {t_new}.");
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def MejorOR(solucion, surgeon, second, OT, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);
    scheduled = [p for p in range(len(pacientes_copy)) if pacientes_copy[p] != -1];

    if not scheduled:
        print("[MejorOR] No hay pacientes programados.") if hablar else None;
        return solucion;

    feasible_moves = [];
    for p in scheduled:
        start_blk = pacientes_copy[p];
        o_old, d_old, t_old = decompress(start_blk, nSlot, nDays);
        s = primarios_copy[start_blk];
        a = secundarios_copy[start_blk];
        dur = OT[p];

        for o_new in range(len(or_schedule_copy)):
            if o_new == o_old:
                continue;

            new_start = compress(o_new, d_old, t_old, nSlot, nDays);

            # Check if there are enough fichas
            if fichas_copy[(s, d_old)] < dictCosts[(s, a, new_start)]:
                continue;

            can_move = True;
            for b in range(dur):
                if AOR[p][o_new][t_old + b][d_old % 5] != 1:
                    can_move = False;
                    break;

                new_blk = compress(o_new, d_old, t_old + b, nSlot, nDays);
                if new_blk in primarios_copy or new_blk in secundarios_copy:
                    can_move = False;
                    break;

            if can_move:
                feasible_moves.append((p, o_old, d_old, t_old, o_new, s, a));

    if not feasible_moves:
        print("[MejorOR] No hay cambios realizables.") if hablar else None;
        return solucion;

    p_sel, o_old, d_old, t_old, o_new, s, a = random.choice(feasible_moves);
    dur = OT[p_sel];
    start_blk_old = pacientes_copy[p_sel];

    # Remove from old position
    for b in range(dur):
        ob = start_blk_old + b;
        if ob in primarios_copy:
            surgeon_schedule_copy[primarios_copy[ob]][d_old][t_old + b] = -1;
            del primarios_copy[ob];
        if ob in secundarios_copy:
            surgeon_schedule_copy[secundarios_copy[ob]][d_old][t_old + b] = -1;
            del secundarios_copy[ob];
        or_schedule_copy[o_old][d_old][t_old + b] = -1;

    # Add to new position
    new_start = compress(o_new, d_old, t_old, nSlot, nDays);

    # Update fichas for the surgeon
    fichas_copy[(s, d_old)] -= dictCosts[(s, a, new_start)];

    for b in range(dur):
        primarios_copy[new_start + b] = s;
        secundarios_copy[new_start + b] = a;
        surgeon_schedule_copy[s][d_old][t_old + b] = p_sel;
        surgeon_schedule_copy[a][d_old][t_old + b] = p_sel;
        or_schedule_copy[o_new][d_old][t_old + b] = p_sel;
    pacientes_copy[p_sel] = new_start;

    if hablar:
        print(f"[MejorOR] Patient {p_sel} moved from OR {o_old} to OR {o_new}, day={d_old}, slot={t_old}.");

    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy);