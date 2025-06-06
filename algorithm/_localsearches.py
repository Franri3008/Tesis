import random
import copy
import math

def compress(o, d, t, nSlot, nDays):
    return o * nSlot * nDays + d * nSlot + t

def decompress(val, nSlot, nDays):
    o = val // (nSlot * nDays)
    temp = val % (nSlot * nDays)
    d = temp // nSlot
    t = temp % nSlot
    return o, d, t

def is_feasible_block(p, o, d, t, AOR):
    return AOR[p][o][t][d % 5] == 1

def MejorarAfinidad_primario(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
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

def MejorarAfinidad_secundario(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
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

def AdelantarDia(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
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
                    if not all(fichas_copy[(s,d_aux)] - dictCosts[(s,a,new_blk)] >= 0 for d_aux in range(new_d, nDays)):
                        continue
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
    old_cost = dictCosts[(s, a, start_blk_old)]
    new_cost = dictCosts[(s, a, new_start)]
    for d_aux in range(d_old, nDays):
        fichas_copy[(s, d_aux)] += old_cost
    for d_aux in range(d_new, nDays):
        fichas_copy[(s, d_aux)] -= new_cost 
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

def MejorOR(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surgeon_schedule_copy = copy.deepcopy(solucion[1])
    or_schedule_copy = copy.deepcopy(solucion[2])
    fichas_copy = copy.deepcopy(solucion[3])
    pacientes_copy = copy.deepcopy(solucion[0][0])
    primarios_copy = copy.deepcopy(solucion[0][1])
    secundarios_copy = copy.deepcopy(solucion[0][2])
    scheduled = [p for p in range(len(pacientes_copy)) if pacientes_copy[p] != -1]

    if not scheduled:
        if hablar:
            print("[MejorOR] No hay pacientes programados.")
        return solucion

    best_move = None
    max_new_cost = -math.inf
    for p in scheduled:
        start_blk_old = pacientes_copy[p]
        if start_blk_old == -1:
             continue
        o_old, d_old, t_old = decompress(start_blk_old, nSlot, nDays)
        if start_blk_old not in primarios_copy or start_blk_old not in secundarios_copy:
            print(f"[MejorOR] Warning: Inconsistent state for patient {p} at block {start_blk_old}. Skipping.")
            continue
        s = primarios_copy[start_blk_old]
        a = secundarios_copy[start_blk_old]
        dur = OT[p]
        current_cost_key = (s, a, start_blk_old)
        if current_cost_key not in dictCosts:
            print(f"[MejorOR] Warning: Cost not found for current assignment {current_cost_key}. Skipping patient {p}.")
            continue
        current_cost = dictCosts[current_cost_key]
        for o_new in range(len(or_schedule_copy)):
            if o_new == o_old:
                continue
            new_start = compress(o_new, d_old, t_old, nSlot, nDays)
            new_cost_key = (s, a, new_start)
            if new_cost_key not in dictCosts:
                continue # Cannot evaluate if cost is undefined
            new_cost = dictCosts[new_cost_key]
            if not all(fichas_copy[(s, d_aux)] + dictCosts[(s, a, start_blk_old)] - dictCosts[(s, a, new_start)] >= 0 for d_aux in range(d_old, nDays)) or fichas_copy[(s, d_old)] + current_cost < new_cost:
                continue;
            can_move = True
            for b in range(dur):
                if t_old + b >= nSlot or AOR[p][o_new][t_old + b][d_old%5] != 1:
                    can_move = False
                    break

                new_blk = compress(o_new, d_old, t_old + b, nSlot, nDays)
                if new_blk in primarios_copy or new_blk in secundarios_copy:
                     can_move = False
                     break

            if can_move:
                if new_cost > max_new_cost:
                    max_new_cost = new_cost
                    best_move = (p, o_old, d_old, t_old, o_new, s, a, dur, start_blk_old, current_cost, new_start, new_cost)
    if best_move is None:
        if hablar:
            print("[MejorOR_HighestCost] No feasible OR change found.")
        return solucion # Return original solution if no feasible move found
    p_sel, o_old, d_old, t_old, o_new, s, a, dur, start_blk_old, old_cost_sel, new_start_sel, new_cost_sel = best_move

    if hablar:
        print(f"[MejorOR_HighestCost] Found highest-cost feasible move: Patient {p_sel} from OR {o_old} to OR {o_new} (Day {d_old}, Slot {t_old}). Cost change: {old_cost_sel} -> {new_cost_sel}")

    for b in range(dur):
        ob = start_blk_old + b
        primarios_copy.pop(ob, None)
        secundarios_copy.pop(ob, None)
        if 0 <= o_old < len(or_schedule_copy) and 0 <= d_old < len(or_schedule_copy[o_old]) and 0 <= t_old + b < len(or_schedule_copy[o_old][d_old]):
             or_schedule_copy[o_old][d_old][t_old + b] = -1
        if 0 <= s < len(surgeon_schedule_copy) and 0 <= d_old < len(surgeon_schedule_copy[s]) and 0 <= t_old + b < len(surgeon_schedule_copy[s][d_old]):
             if surgeon_schedule_copy[s][d_old][t_old + b] == p_sel:
                 surgeon_schedule_copy[s][d_old][t_old + b] = -1
        if 0 <= a < len(surgeon_schedule_copy) and 0 <= d_old < len(surgeon_schedule_copy[a]) and 0 <= t_old + b < len(surgeon_schedule_copy[a][d_old]):
             if surgeon_schedule_copy[a][d_old][t_old + b] == p_sel:
                 surgeon_schedule_copy[a][d_old][t_old + b] = -1

    cost_diff = old_cost_sel - new_cost_sel
    for d_aux in range(d_old, nDays):
        if (s, d_aux) in fichas_copy:
            fichas_copy[(s, d_aux)] += cost_diff

    pacientes_copy[p_sel] = new_start_sel
    for b in range(dur):
        new_blk = new_start_sel + b
        primarios_copy[new_blk] = s
        secundarios_copy[new_blk] = a
        if 0 <= o_new < len(or_schedule_copy) and 0 <= d_old < len(or_schedule_copy[o_new]) and 0 <= t_old + b < len(or_schedule_copy[o_new][d_old]):
             or_schedule_copy[o_new][d_old][t_old + b] = p_sel
        if 0 <= s < len(surgeon_schedule_copy) and 0 <= d_old < len(surgeon_schedule_copy[s]) and 0 <= t_old + b < len(surgeon_schedule_copy[s][d_old]):
             surgeon_schedule_copy[s][d_old][t_old + b] = p_sel
        if 0 <= a < len(surgeon_schedule_copy) and 0 <= d_old < len(surgeon_schedule_copy[a]) and 0 <= t_old + b < len(surgeon_schedule_copy[a][d_old]):
             surgeon_schedule_copy[a][d_old][t_old + b] = p_sel

    if hablar:
        print(f"[MejorOR_HighestCost] Patient {p_sel} successfully moved from OR {o_old} to OR {o_new}, day={d_old}, slot={t_old}. New cost is {new_cost_sel}.")
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def AdelantarTodos(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surg_sched = copy.deepcopy(solucion[1]);
    or_sched = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    pacientes, prim, sec = (copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]));
    num_ors = len(or_sched);
    scheduled = [p for p,v in enumerate(pacientes) if v != -1];
    if not scheduled: return solucion;

    for p in scheduled:
        blk_old = pacientes[p]; o_old,d_old,t_old = decompress(blk_old,nSlot,nDays);
        if d_old == 0 or blk_old not in prim or blk_old not in sec: continue;
        s = prim[blk_old]; a = sec[blk_old]; dur = OT[p];
        old_cost = dictCosts[(s,a,blk_old)];

        best = None
        for new_d in range(d_old):
            for new_t in range(nSlot-dur+1):
                if new_t < nSlot//2 < new_t+dur: continue
                for new_o in range(num_ors):
                    if AOR[p][new_o][new_t][new_d%5] != 1: continue
                    blk_new = compress(new_o,new_d,new_t,nSlot,nDays)
                    new_cost = dictCosts[(s,a,blk_new)]
                    can_afford = True
                    for d_check in range(new_d, nDays):
                        if d_check < d_old:
                            delta = new_cost
                        else:
                            delta = new_cost - old_cost
                        if fichas_copy[(s, d_check)] + (-delta) < 0:
                            can_afford = False
                            break
                    if not can_afford:
                        continue
                    # surgeons free?
                    if any(surg_sched[s][new_d][new_t+b] != -1 for b in range(dur)): continue
                    if any(surg_sched[a][new_d][new_t+b] != -1 for b in range(dur)): continue
                    # OR blocks free?
                    if any(compress(new_o,new_d,new_t+b,nSlot,nDays) in prim
                            or compress(new_o,new_d,new_t+b,nSlot,nDays) in sec
                            or or_sched[new_o][new_d][new_t+b] != -1
                            for b in range(dur)): continue
                    if best is None or new_d < best[1] or (new_d==best[1] and new_t < best[2]):
                        best = (blk_new,new_d,new_t,new_o,new_cost)
        if best is None: continue
        blk_new,d_new,t_new,o_new,new_cost = best
        for b in range(dur):
            blk=blk_old+b; prim.pop(blk,None); sec.pop(blk,None)
            or_sched[o_old][d_old][t_old+b] = -1
            if surg_sched[s][d_old][t_old+b]==p: surg_sched[s][d_old][t_old+b]=-1
            if surg_sched[a][d_old][t_old+b]==p: surg_sched[a][d_old][t_old+b]=-1

        for d_ in range(d_old,nDays): fichas_copy[(s,d_)] += old_cost
        for d_ in range(d_new,nDays): fichas_copy[(s,d_)] -= new_cost

        pacientes[p] = blk_new
        for b in range(dur):
            blk=blk_new+b; prim[blk]=s; sec[blk]=a
            or_sched[o_new][d_new][t_new+b]=p
            surg_sched[s][d_new][t_new+b]=p
            surg_sched[a][d_new][t_new+b]=p

        if hablar:
            print(f"[AdelantarTodos] p{p} Day {d_old}->{d_new}, OR {o_old}->{o_new}")

    return ((pacientes,prim,sec),surg_sched,or_sched,fichas_copy)

def CambiarPaciente1(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surgeon_schedule_copy = copy.deepcopy(solucion[1])
    or_schedule_copy = copy.deepcopy(solucion[2])
    fichas_copy = copy.deepcopy(solucion[3])
    pacientes_copy = copy.deepcopy(solucion[0][0])
    primarios_copy = copy.deepcopy(solucion[0][1])
    secundarios_copy = copy.deepcopy(solucion[0][2])
    num_patients = len(pacientes_copy)
    num_ors = len(or_schedule_copy);

    unscheduled_patients = [p for p in range(num_patients) if pacientes_copy[p] == -1]
    scheduled_patients = [p for p in range(num_patients) if pacientes_copy[p] != -1]

    if not unscheduled_patients or not scheduled_patients:
        if hablar:
            print("[CambiarPaciente1] No unscheduled or scheduled patients available for swap.")
        return solucion

    swap_to_perform = None
    for p_unsched in unscheduled_patients:
        dur_unsched = OT[p_unsched]
        if swap_to_perform:
             break
        for p_sched in scheduled_patients:
            start_blk_sched = pacientes_copy[p_sched]
            if start_blk_sched == -1:
                continue
            if start_blk_sched not in primarios_copy or start_blk_sched not in secundarios_copy:
                 if hablar:
                     print(f"[CambiarPaciente1] Warning: Inconsistent state for scheduled patient {p_sched} at block {start_blk_sched}. Skipping.")
                 continue

            o, d, t = decompress(start_blk_sched, nSlot, nDays)
            s = primarios_copy[start_blk_sched];
            a = secundarios_copy[start_blk_sched];
            dur_sched = OT[p_sched]

            if t + dur_unsched > nSlot:
                continue
            if t < nSlot // 2 and t + dur_unsched > nSlot // 2:
                continue

            can_place_unsched = True
            for b in range(dur_unsched):
                current_t = t + b
                or_val = or_schedule_copy[o][d][current_t]
                if not (or_val == -1 or or_val == p_sched):
                    can_place_unsched = False; break
                s_val = surgeon_schedule_copy[s][d][current_t]
                if not (s_val == -1 or s_val == p_sched):
                    can_place_unsched = False; break
                a_val = surgeon_schedule_copy[a][d][current_t]
                if not (a_val == -1 or a_val == p_sched):
                    can_place_unsched = False; break
                if AOR[p_unsched][o][current_t][d % 5] != 1:
                    can_place_unsched = False; break

            if not can_place_unsched:
                continue
            if SP[p_unsched][s] != 1:
                continue

            cost_key = (s, a, start_blk_sched);
            if cost_key not in dictCosts:
                 if hablar:
                     print(f"[CambiarPaciente1] Warning: Cost key {cost_key} not found for scheduled patient {p_sched}'s slot. Skipping swap.")
                 continue
            cost_sched = dictCosts[cost_key];
            cost_unsched = dictCosts[cost_key];

            fichas_ok = True
            cost_diff = cost_sched - cost_unsched;
            for d_aux in range(d, nDays):
                current_fichas = fichas_copy.get((s, d_aux), -math.inf);
                if current_fichas + cost_diff < 0:
                    fichas_ok = False
                    break
            if not fichas_ok:
                continue
            swap_to_perform = (p_unsched, p_sched, start_blk_sched, dur_sched, dur_unsched, s, a, cost_diff)
            if hablar:
                print(f"[CambiarPaciente1] Found feasible swap: Unscheduled {p_unsched}({dur_unsched}) <-> Scheduled {p_sched}({dur_sched}) at slot ({o},{d},{t})")
            break

    if swap_to_perform:
        p_unsched_sel, p_sched_sel, start_blk, dur_sched_orig, dur_unsched_new, s_sel, a_sel, cost_difference = swap_to_perform
        o, d, t = decompress(start_blk, nSlot, nDays)

        for d_aux in range(d, nDays):
            fichas_key = (s_sel, d_aux)
            if fichas_key in fichas_copy:
                fichas_copy[fichas_key] += cost_difference
        for b in range(dur_sched_orig):
            blk_old = start_blk + b
            current_t = t + b
            if 0 <= o < num_ors and 0 <= d < nDays and 0 <= current_t < nSlot:
                if or_schedule_copy[o][d][current_t] == p_sched_sel:
                    or_schedule_copy[o][d][current_t] = -1
                if surgeon_schedule_copy[s_sel][d][current_t] == p_sched_sel:
                    surgeon_schedule_copy[s_sel][d][current_t] = -1
                if surgeon_schedule_copy[a_sel][d][current_t] == p_sched_sel:
                    surgeon_schedule_copy[a_sel][d][current_t] = -1
            primarios_copy.pop(blk_old, None);
            secundarios_copy.pop(blk_old, None);

        for b in range(dur_unsched_new):
            blk_new = start_blk + b;
            current_t = t + b
            if 0 <= o < num_ors and 0 <= d < nDays and 0 <= current_t < nSlot:
                or_schedule_copy[o][d][current_t] = p_unsched_sel
                surgeon_schedule_copy[s_sel][d][current_t] = p_unsched_sel
                surgeon_schedule_copy[a_sel][d][current_t] = p_unsched_sel
                primarios_copy[blk_new] = s_sel
                secundarios_copy[blk_new] = a_sel
            else:
                print(f"[CambiarPaciente1] Error: Trying to assign block outside schedule bounds during execution ({o},{d},{current_t}).") if hablar else None;
        pacientes_copy[p_sched_sel] = -1;
        pacientes_copy[p_unsched_sel] = start_blk;

        if hablar:
            print(f"[CambiarPaciente1] Executed swap: Patient {p_unsched_sel} ({dur_unsched_new} blocks) now in slot starting ({o},{d},{t}). Patient {p_sched_sel} unscheduled.")

        return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)
    else:
        if hablar:
            print("[CambiarPaciente1] No feasible swap found.")
        return solucion

def CambiarPaciente2(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surgeon_schedule_copy = copy.deepcopy(solucion[1])
    or_schedule_copy = copy.deepcopy(solucion[2])
    fichas_copy = copy.deepcopy(solucion[3])
    pacientes_copy = copy.deepcopy(solucion[0][0])
    primarios_copy = copy.deepcopy(solucion[0][1])
    secundarios_copy = copy.deepcopy(solucion[0][2])
    num_patients = len(pacientes_copy)
    num_ors = len(or_schedule_copy)

    unscheduled_patients = [p for p in range(num_patients) if pacientes_copy[p] == -1]
    scheduled_patients = [p for p in range(num_patients) if pacientes_copy[p] != -1]

    if not unscheduled_patients or not scheduled_patients:
        if hablar:
            print("[CambiarPaciente2] No unscheduled or scheduled patients available for swap.")
        return solucion

    best_swap_found = None
    max_priority_ratio = -math.inf;
    for p_unsched in unscheduled_patients:
        dur_unsched = OT[p_unsched]
        if dur_unsched <= 0:
            if hablar: print(f"[CambiarPaciente2] Skipping unscheduled patient {p_unsched} with non-positive duration {dur_unsched}.")
            continue

        for p_sched in scheduled_patients:
            start_blk_sched = pacientes_copy[p_sched]
            if start_blk_sched == -1: continue

            if start_blk_sched not in primarios_copy or start_blk_sched not in secundarios_copy:
                 if hablar: print(f"[CambiarPaciente2] Warning: Inconsistent state for scheduled patient {p_sched} at block {start_blk_sched}. Skipping.")
                 continue

            o, d, t = decompress(start_blk_sched, nSlot, nDays)
            s = primarios_copy[start_blk_sched]
            a = secundarios_copy[start_blk_sched]
            dur_sched = OT[p_sched]
            if t + dur_unsched > nSlot: continue
            can_place_unsched = True
            for b in range(dur_unsched):
                current_t = t + b
                if not (0 <= o < num_ors and 0 <= d < nDays and 0 <= current_t < nSlot):
                    can_place_unsched = False; break
                or_val = or_schedule_copy[o][d][current_t]
                if not (or_val == -1 or or_val == p_sched): can_place_unsched = False; break
                s_val = surgeon_schedule_copy[s][d][current_t]
                if not (s_val == -1 or s_val == p_sched): can_place_unsched = False; break
                a_val = surgeon_schedule_copy[a][d][current_t]
                if not (a_val == -1 or a_val == p_sched): can_place_unsched = False; break
                if AOR[p_unsched][o][current_t][d % 5] != 1: can_place_unsched = False; break
            if not can_place_unsched: continue
            if SP[p_unsched][s] != 1: continue
            cost_key = (s, a, start_blk_sched)
            if cost_key not in dictCosts:
                 if hablar: print(f"[CambiarPaciente2] Warning: Cost key {cost_key} not found for slot. Skipping swap.")
                 continue
            cost_sched = dictCosts[cost_key]
            cost_unsched = dictCosts[cost_key]
            cost_diff = cost_sched - cost_unsched

            fichas_ok = True
            for d_aux in range(d, nDays):
                current_fichas = fichas_copy.get((s, d_aux), -math.inf)
                if current_fichas + cost_diff < 0:
                    fichas_ok = False; break
            if not fichas_ok: continue
            current_priority = I[(p_unsched, 0)];
            current_ratio = current_priority / dur_unsched
            if hablar:
                 print(f"[CambiarPaciente2] Feasible swap found: Unscheduled {p_unsched}(P={current_priority},D={dur_unsched},R={current_ratio:.2f}) <-> Scheduled {p_sched} at ({o},{d},{t}). Current max ratio: {max_priority_ratio:.2f}")

            if current_ratio > max_priority_ratio:
                max_priority_ratio = current_ratio
                best_swap_found = (p_unsched, p_sched, start_blk_sched, dur_sched, dur_unsched, s, a, cost_diff)
                if hablar:
                    print(f"[CambiarPaciente2] ---> New best swap found with ratio {max_priority_ratio:.2f}")

    if best_swap_found:
        p_unsched_sel, p_sched_sel, start_blk, dur_sched_orig, dur_unsched_new, s_sel, a_sel, cost_difference = best_swap_found
        o, d, t = decompress(start_blk, nSlot, nDays)

        if hablar:
            print(f"[CambiarPaciente2] Executing best swap: Unscheduled {p_unsched_sel} (Ratio: {max_priority_ratio:.2f}) replacing Scheduled {p_sched_sel} at ({o},{d},{t})")
        for d_aux in range(d, nDays):
            fichas_key = (s_sel, d_aux)
            if fichas_key in fichas_copy:
                fichas_copy[fichas_key] += cost_difference
        for b in range(dur_sched_orig):
            blk_old = start_blk + b
            current_t = t + b
            if 0 <= o < num_ors and 0 <= d < nDays and 0 <= current_t < nSlot:
                if or_schedule_copy[o][d][current_t] == p_sched_sel: or_schedule_copy[o][d][current_t] = -1
                if surgeon_schedule_copy[s_sel][d][current_t] == p_sched_sel: surgeon_schedule_copy[s_sel][d][current_t] = -1
                if surgeon_schedule_copy[a_sel][d][current_t] == p_sched_sel: surgeon_schedule_copy[a_sel][d][current_t] = -1
            primarios_copy.pop(blk_old, None)
            secundarios_copy.pop(blk_old, None)
        for b in range(dur_unsched_new):
            blk_new = start_blk + b
            current_t = t + b
            if 0 <= o < num_ors and 0 <= d < nDays and 0 <= current_t < nSlot:
                or_schedule_copy[o][d][current_t] = p_unsched_sel
                surgeon_schedule_copy[s_sel][d][current_t] = p_unsched_sel
                surgeon_schedule_copy[a_sel][d][current_t] = p_unsched_sel
                primarios_copy[blk_new] = s_sel
                secundarios_copy[blk_new] = a_sel
            else:
                 if hablar: print(f"[CambiarPaciente2] Error: Trying to assign block outside bounds during execution ({o},{d},{current_t}).")
        pacientes_copy[p_sched_sel] = -1
        pacientes_copy[p_unsched_sel] = start_blk

        return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)
    else:
        if hablar:
            print("[CambiarPaciente2] No feasible swap found after checking all possibilities.")
        return solucion

def CambiarPaciente3(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surgeon_schedule_copy = copy.deepcopy(solucion[1])
    or_schedule_copy = copy.deepcopy(solucion[2])
    fichas_copy = copy.deepcopy(solucion[3])
    pacientes_copy = copy.deepcopy(solucion[0][0])
    primarios_copy = copy.deepcopy(solucion[0][1])
    secundarios_copy = copy.deepcopy(solucion[0][2])
    num_patients = len(pacientes_copy)
    num_ors = len(or_schedule_copy)
    initial_unscheduled_patients = [p for p in range(num_patients) if pacientes_copy[p] == -1]

    if not initial_unscheduled_patients:
        if hablar:
            print("[CambiarPaciente3] No unscheduled patients to attempt swapping.")
        return solucion

    made_a_swap_overall = False
    for p_unsched in initial_unscheduled_patients:
        if pacientes_copy[p_unsched] != -1:
            continue

        dur_unsched = OT[p_unsched]
        if dur_unsched <= 0: continue

        found_swap_for_this_patient = None
        for p_sched_idx in range(num_patients):
            start_blk_sched = pacientes_copy[p_sched_idx]
            if start_blk_sched == -1:
                continue

            p_sched = p_sched_idx
            if start_blk_sched not in primarios_copy or start_blk_sched not in secundarios_copy:
                 if hablar: print(f"[CambiarPaciente3] Warning: Inconsistent state for scheduled patient {p_sched} at block {start_blk_sched}. Skipping.")
                 continue

            o, d, t = decompress(start_blk_sched, nSlot, nDays)
            s = primarios_copy[start_blk_sched]
            a = secundarios_copy[start_blk_sched]
            dur_sched = OT[p_sched]

            if t + dur_unsched > nSlot: continue
            can_place_unsched = True
            for b in range(dur_unsched):
                current_t = t + b
                if not (0 <= o < num_ors and 0 <= d < nDays and 0 <= current_t < nSlot): can_place_unsched = False; break
                or_val = or_schedule_copy[o][d][current_t]
                if not (or_val == -1 or or_val == p_sched): can_place_unsched = False; break
                s_val = surgeon_schedule_copy[s][d][current_t]
                if not (s_val == -1 or s_val == p_sched): can_place_unsched = False; break
                a_val = surgeon_schedule_copy[a][d][current_t]
                if not (a_val == -1 or a_val == p_sched): can_place_unsched = False; break
                if AOR[p_unsched][o][current_t][d % 5] != 1: can_place_unsched = False; break
            if not can_place_unsched: continue
            if SP[p_unsched][s] != 1: continue

            cost_key = (s, a, start_blk_sched)
            if cost_key not in dictCosts:
                 if hablar: print(f"[CambiarPaciente3] Warning: Cost key {cost_key} not found. Skipping.")
                 continue
            cost_sched = dictCosts[cost_key]
            cost_unsched = dictCosts[cost_key]
            cost_diff = cost_sched - cost_unsched

            fichas_ok = True
            for d_aux in range(d, nDays):
                current_fichas = fichas_copy.get((s, d_aux), -math.inf)
                if current_fichas + cost_diff < 0:
                    fichas_ok = False; break
            if not fichas_ok: continue
            found_swap_for_this_patient = (p_sched, start_blk_sched, dur_sched, dur_unsched, s, a, cost_diff)
            if hablar:
                print(f"[CambiarPaciente3] Found feasible swap for unscheduled {p_unsched}: Replacing scheduled {p_sched} at ({o},{d},{t})")
            break

        if found_swap_for_this_patient:
            made_a_swap_overall = True
            p_sched_sel, start_blk, dur_sched_orig, dur_unsched_new, s_sel, a_sel, cost_difference = found_swap_for_this_patient
            o, d, t = decompress(start_blk, nSlot, nDays)

            if hablar:
                print(f"[CambiarPaciente3] --> Executing swap: {p_unsched}({dur_unsched_new}) replaces {p_sched_sel}({dur_sched_orig}) at ({o},{d},{t})")
            for d_aux in range(d, nDays):
                fichas_key = (s_sel, d_aux)
                if fichas_key in fichas_copy:
                    fichas_copy[fichas_key] += cost_difference

            for b in range(dur_sched_orig):
                blk_old = start_blk + b
                current_t = t + b
                if 0 <= o < num_ors and 0 <= d < nDays and 0 <= current_t < nSlot:
                    if or_schedule_copy[o][d][current_t] == p_sched_sel: or_schedule_copy[o][d][current_t] = -1
                    if surgeon_schedule_copy[s_sel][d][current_t] == p_sched_sel: surgeon_schedule_copy[s_sel][d][current_t] = -1
                    if surgeon_schedule_copy[a_sel][d][current_t] == p_sched_sel: surgeon_schedule_copy[a_sel][d][current_t] = -1
                primarios_copy.pop(blk_old, None)
                secundarios_copy.pop(blk_old, None)
            for b in range(dur_unsched_new):
                blk_new = start_blk + b
                current_t = t + b
                if 0 <= o < num_ors and 0 <= d < nDays and 0 <= current_t < nSlot:
                    or_schedule_copy[o][d][current_t] = p_unsched
                    surgeon_schedule_copy[s_sel][d][current_t] = p_unsched
                    surgeon_schedule_copy[a_sel][d][current_t] = p_unsched
                    primarios_copy[blk_new] = s_sel
                    secundarios_copy[blk_new] = a_sel
                else:
                     if hablar: print(f"[CambiarPaciente3] Error: Assigning block outside bounds ({o},{d},{current_t}).")
            pacientes_copy[p_sched_sel] = -1
            pacientes_copy[p_unsched] = start_blk
    if not made_a_swap_overall and hablar:
        print("[CambiarPaciente3] No feasible swaps could be made for any unscheduled patient.")
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def CambiarPaciente4(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    pacientes_copy = copy.deepcopy(solucion[0][0]);
    primarios_copy = copy.deepcopy(solucion[0][1]);
    secundarios_copy = copy.deepcopy(solucion[0][2]);
    num_patients = len(pacientes_copy);
    num_ors = len(or_schedule_copy);

    unscheduled = [p for p in range(num_patients) if pacientes_copy[p] == -1 and OT[p] > 0];
    scheduled = [p for p in range(num_patients) if pacientes_copy[p] != -1 and OT[p] > 0];

    if not unscheduled or not scheduled:
        return solucion;

    p_unsched = max(unscheduled, key=lambda p: I[(p,0)]/OT[p]);
    dur_unsched = OT[p_unsched];
    best_target = None;
    min_ratio = math.inf;

    for p_sched in scheduled:
        start_blk = pacientes_copy[p_sched];
        if start_blk not in primarios_copy or start_blk not in secundarios_copy:
            continue;
        o,d,t = decompress(start_blk,nSlot,nDays);
        s = primarios_copy[start_blk];
        a = secundarios_copy[start_blk];
        if SP[p_unsched][s] != 1:
            continue;
        if t+dur_unsched>nSlot or (t<nSlot//2 and t+dur_unsched>nSlot//2):
            continue;
        can_place = True;
        for b in range(dur_unsched):
            cur_t=t+b;
            if AOR[p_unsched][o][cur_t][d%5]!=1:
                can_place=False;break;
            if surgeon_schedule_copy[s][d][cur_t] not in (-1,p_sched):
                can_place=False;break;
            if surgeon_schedule_copy[a][d][cur_t] not in (-1,p_sched):
                can_place=False;break;
            if or_schedule_copy[o][d][cur_t] not in (-1,p_sched):
                can_place=False;break;
        if not can_place:
            continue;
        cost_key=(s,a,start_blk);
        if cost_key not in dictCosts:
            continue;
        cost_diff=0;
        for d_aux in range(d,nDays):
            if fichas_copy.get((s,d_aux),-math.inf)+cost_diff<0:
                can_place=False;break;
        if not can_place:
            continue;
        ratio_sched=I[(p_sched,0)]/OT[p_sched];
        if ratio_sched<min_ratio:
            min_ratio=ratio_sched;
            best_target=(p_sched,start_blk,dur_unsched,OT[p_sched],s,a,cost_diff);

    if best_target is None:
        return solucion;

    p_sched_sel,start_blk,dur_unsched_new,dur_sched_old,s_sel,a_sel,cost_diff=best_target;
    o,d,t=decompress(start_blk,nSlot,nDays);

    for d_aux in range(d,nDays):
        fichas_copy[(s_sel,d_aux)]+=cost_diff;

    for b in range(dur_sched_old):
        blk_old=start_blk+b;
        cur_t=t+b;
        primarios_copy.pop(blk_old,None);
        secundarios_copy.pop(blk_old,None);
        or_schedule_copy[o][d][cur_t]=-1;
        surgeon_schedule_copy[s_sel][d][cur_t]=-1 if surgeon_schedule_copy[s_sel][d][cur_t]==p_sched_sel else surgeon_schedule_copy[s_sel][d][cur_t];
        surgeon_schedule_copy[a_sel][d][cur_t]=-1 if surgeon_schedule_copy[a_sel][d][cur_t]==p_sched_sel else surgeon_schedule_copy[a_sel][d][cur_t];

    for b in range(dur_unsched_new):
        blk_new=start_blk+b;
        cur_t=t+b;
        primarios_copy[blk_new]=s_sel;
        secundarios_copy[blk_new]=a_sel;
        or_schedule_copy[o][d][cur_t]=p_unsched;
        surgeon_schedule_copy[s_sel][d][cur_t]=p_unsched;
        surgeon_schedule_copy[a_sel][d][cur_t]=p_unsched;

    pacientes_copy[p_sched_sel]=-1;
    pacientes_copy[p_unsched]=start_blk;

    return ((pacientes_copy,primarios_copy,secundarios_copy),surgeon_schedule_copy,or_schedule_copy,fichas_copy);

def CambiarPaciente5(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surgeon_schedule_copy=copy.deepcopy(solucion[1]);
    or_schedule_copy=copy.deepcopy(solucion[2]);
    fichas_copy=copy.deepcopy(solucion[3]);
    pacientes_copy=copy.deepcopy(solucion[0][0]);
    primarios_copy=copy.deepcopy(solucion[0][1]);
    secundarios_copy=copy.deepcopy(solucion[0][2]);
    num_patients=len(pacientes_copy);
    num_ors=len(or_schedule_copy);

    unscheduled=[p for p in range(num_patients) if pacientes_copy[p]==-1 and OT[p]>0];
    scheduled=[p for p in range(num_patients) if pacientes_copy[p]!=-1 and OT[p]>0];
    if not unscheduled or not scheduled:
        return solucion;

    best=None;
    best_score=-math.inf;
    for p_u in unscheduled:
        dur_u=OT[p_u];
        ratio_u=I[(p_u,0)]/dur_u;
        for p_s in scheduled:
            start_blk=pacientes_copy[p_s];
            if start_blk not in primarios_copy or start_blk not in secundarios_copy:
                continue;
            o,d,t=decompress(start_blk,nSlot,nDays);
            s=primarios_copy[start_blk];
            a=secundarios_copy[start_blk];
            dur_s=OT[p_s];
            if SP[p_u][s]!=1:
                continue;
            if t+dur_u>nSlot or (t<nSlot//2 and t+dur_u>nSlot//2):
                continue;
            feasible=True;
            for b in range(dur_u):
                ct=t+b;
                if AOR[p_u][o][ct][d%5]!=1: feasible=False;break;
                if or_schedule_copy[o][d][ct] not in(-1,p_s): feasible=False;break;
                if surgeon_schedule_copy[s][d][ct] not in(-1,p_s): feasible=False;break;
                if surgeon_schedule_copy[a][d][ct] not in(-1,p_s): feasible=False;break;
            if not feasible:
                continue;
            cost_key=(s,a,start_blk);
            if cost_key not in dictCosts:
                continue;
            cost_diff=0;
            fichas_ok=True;
            for d_aux in range(d,nDays):
                if fichas_copy.get((s,d_aux),-math.inf)+cost_diff<0:
                    fichas_ok=False;break;
            if not fichas_ok:
                continue;
            ratio_s=I[(p_s,0)]/dur_s;
            score=ratio_u-ratio_s;
            if score>best_score and score>0:
                best_score=score;
                best=(p_u,p_s,start_blk,dur_u,dur_s,s,a,cost_diff);
    if best is None:
        return solucion;

    p_u,p_s,start_blk,dur_u,dur_s,s_sel,a_sel,cost_diff=best;
    o,d,t=decompress(start_blk,nSlot,nDays);
    for d_aux in range(d,nDays):
        fichas_copy[(s_sel,d_aux)]+=cost_diff;
    for b in range(dur_s):
        blk_old=start_blk+b;
        ct=t+b;
        or_schedule_copy[o][d][ct]=-1;
        if surgeon_schedule_copy[s_sel][d][ct]==p_s: surgeon_schedule_copy[s_sel][d][ct]=-1;
        if surgeon_schedule_copy[a_sel][d][ct]==p_s: surgeon_schedule_copy[a_sel][d][ct]=-1;
        primarios_copy.pop(blk_old,None);
        secundarios_copy.pop(blk_old,None);
    for b in range(dur_u):
        blk_new=start_blk+b;
        ct=t+b;
        or_schedule_copy[o][d][ct]=p_u;
        surgeon_schedule_copy[s_sel][d][ct]=p_u;
        surgeon_schedule_copy[a_sel][d][ct]=p_u;
        primarios_copy[blk_new]=s_sel;
        secundarios_copy[blk_new]=a_sel;
    pacientes_copy[p_s]=-1;
    pacientes_copy[p_u]=start_blk;
    return((pacientes_copy,primarios_copy,secundarios_copy),surgeon_schedule_copy,or_schedule_copy,fichas_copy);

