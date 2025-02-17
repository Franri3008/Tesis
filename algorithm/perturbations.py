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


##################################
######### Perturbaciones #########
##################################

def CambiarPrimarios(solucion, surgeon, second, OT, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    #sol = (solucion[0][0].copy(), solucion[0][1].copy(), solucion[0][2].copy());
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    #pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);
    scheduled = [i for i, blk in enumerate(pacientes_copy) if blk != -1]
    if len(scheduled) < 2:
        print("[CambiarPrimarios] No hay suficientes pacientes para el cambio.") if hablar else None;
        return solucion

    valid_pairs = [];
    for i in range(len(scheduled)):
        for j in range(i + 1, len(scheduled)):
            p1 = scheduled[i];
            p2 = scheduled[j];
            start1 = pacientes_copy[p1];
            start2 = pacientes_copy[p2];
            dur1 = OT[p1];
            dur2 = OT[p2];
            o1, d1, t1 = decompress(start1, nSlot, nDays);
            o2, d2, t2 = decompress(start2, nSlot, nDays);
            cir1 = primarios_copy[start1];
            cir2 = primarios_copy[start2];
            sec1 = secundarios_copy[start1];
            sec2 = secundarios_copy[start2];

            can_swap = True;
            # Si hay problemas de compatibilidad:
            if SP[p1][cir2] != 1 or SP[p2][cir1] != 1 or cir1 == sec2 or cir2 == sec1 or cir1 == cir2:
                can_swap = False;
                continue;
            # Si no hay suficientes fichas:
            for d_aux in range(d2, nDays):
                if (dictCosts[(cir1, sec2, start2)] > fichas_copy[(cir1, d_aux)] + dictCosts[(cir1, sec1, start1)] * (d_aux >= d1)):
                    can_swap = False;
                    break;
            for d_aux in range(d1, nDays):
                if (dictCosts[(cir2, sec1, start1)] > fichas_copy[(cir2, d_aux)] + dictCosts[(cir2, sec2, start2)] * (d_aux >= d2)):
                    can_swap = False;
                    break;
            # Si uno de los cirujanos no está disponible para toda la duración de la nueva cirugía:
            if not all(surgeon_schedule_copy[cir2][d1][t1 + b] == -1 for b in range(dur1)):
                can_swap = False;
            if not all(surgeon_schedule_copy[cir1][d2][t2 + b] == -1 for b in range(dur2)):
                can_swap = False;
            
            if can_swap:
                valid_pairs.append((p1, p2));

    if not valid_pairs:
        print("[CambiarPrimarios] No hay parejas válidas para el swap.") if hablar else None;
        return solucion

    p1, p2 = random.choice(valid_pairs);
    start1 = pacientes_copy[p1];
    start2 = pacientes_copy[p2];
    cir1 = primarios_copy[start1];
    cir2 = primarios_copy[start2];
    sec1 = secundarios_copy[start1];
    sec2 = secundarios_copy[start2];
    dur1 = OT[p1];
    dur2 = OT[p2];
    o1, d1, t1 = decompress(start1, nSlot, nDays);
    o2, d2, t2 = decompress(start2, nSlot, nDays);

    for d_aux in range(nDays):
        if d_aux >= d1:
            fichas_copy[(cir1, d_aux)] += dictCosts[(cir1, sec1, start1)];# - dictCosts[(main1, sec2, start2)];
            fichas_copy[(cir2, d_aux)] -= dictCosts[(cir2, sec1, start1)];
        if d_aux >= d2:
            fichas_copy[(cir1, d_aux)] -= dictCosts[(cir1, sec1, start1)];
            fichas_copy[(cir2, d_aux)] += dictCosts[(cir2, sec2, start2)];# - dictCosts[(main2, sec1, start1)];

    print(f"[CambiarPrimarios] Cambiando cirujanos p={p1} ({cir1}) <-> p={p2} ({cir2}).") if hablar else None;
    for b in range(dur1):
        primarios_copy[start1 + b] = cir2;
        surgeon_schedule_copy[cir1][d1][t1 + b] = -1;
        surgeon_schedule_copy[cir2][d1][t1 + b] = p1;
    for b in range(dur2):
        primarios_copy[start2 + b] = cir1;
        surgeon_schedule_copy[cir1][d2][t2 + b] = p2;
        surgeon_schedule_copy[cir2][d2][t2 + b] = -1;
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def CambiarSecundarios(solucion, surgeon, second, OT, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    #sol = (solucion[0][0].copy(), solucion[0][1].copy(), solucion[0][2].copy());
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    #pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);
    scheduled = [i for i, blk in enumerate(pacientes_copy) if blk != -1]
    if len(scheduled) < 2:
        print("[CambiarSecundarios] No hay suficientes pacientes para el cambio.") if hablar else None;
        return solucion
    
    valid_pairs = []
    for i in range(len(scheduled)):
        for j in range(i + 1, len(scheduled)):
            p1 = scheduled[i];
            p2 = scheduled[j];
            blk1 = pacientes_copy[p1];
            blk2 = pacientes_copy[p2];
            o1, d1, t1 = decompress(blk1, nSlot, nDays);
            o2, d2, t2 = decompress(blk2, nSlot, nDays);
            dur1 = OT[p1];
            dur2 = OT[p2];
            sec1 = secundarios_copy[blk1];
            sec2 = secundarios_copy[blk2];
            cir1 = primarios_copy[blk1];
            cir2 = primarios_copy[blk2];
            can_swap = True;
            # Problemas de mismo cirujano
            if cir1 == sec2 or cir2 == sec1 or sec1 == sec2:
                can_swap = False;
            # Si uno de los cirujanos no está disponible para toda la duración de la nueva cirugía
            if not all(surgeon_schedule_copy[sec2][d1][t1 + b] == -1 for b in range(dur1)):
                can_swap = False;
            if not all(surgeon_schedule_copy[sec1][d2][t2 + b] == -1 for b in range(dur2)):
                can_swap = False;
            # Si las fichas del cirujano principal no son suficientes para el cambio
            for d_aux in range(d1, nDays):
                if (dictCosts[(cir1, sec2, blk1)] > fichas_copy[(cir1, d_aux)] + dictCosts[(cir1, sec1, blk1)] * (d_aux >= d1)):
                    can_swap = False;
                    break;
            for d_aux in range(d2, nDays):
                if (dictCosts[(cir2, sec1, blk2)] > fichas_copy[(cir2, d_aux)] + dictCosts[(cir2, sec2, blk2)] * (d_aux >= d2)):
                    can_swap = False;
                    break;
            if can_swap:
                valid_pairs.append((p1, p2))
    if not valid_pairs:
        print("[CambiarSecundarios] No hay parejas válidas para el swap.") if hablar else None;
        return solucion
    
    p1, p2 = random.choice(valid_pairs);
    start1 = pacientes_copy[p1];
    start2 = pacientes_copy[p2];
    cir1 = primarios_copy[start1];
    cir2 = primarios_copy[start2];
    sec1 = secundarios_copy[start1];
    sec2 = secundarios_copy[start2];
    dur1 = OT[p1];
    dur2 = OT[p2];
    o1, d1, t1 = decompress(start1, nSlot, nDays);
    o2, d2, t2 = decompress(start2, nSlot, nDays);

    for d_aux in range(nDays):
        if d_aux >= d1:
            fichas_copy[(cir1, d_aux)] += dictCosts[(cir1, sec1, start1)];
            fichas_copy[(cir1, d_aux)] -= dictCosts[(cir1, sec2, start1)];
        if d_aux >= d2:
            fichas_copy[(cir2, d_aux)] += dictCosts[(cir2, sec2, start2)];
            fichas_copy[(cir2, d_aux)] -= dictCosts[(cir2, sec1, start2)];

    print(f"[CambiarSecundarios] Cambiando cirujanos p={p1} ({sec1}) <-> p={p2} ({sec2}).") if hablar else None;
    for b in range(dur1):
        secundarios_copy[start1 + b] = sec2;
        surgeon_schedule_copy[sec1][d1][t1 + b] = -1;
        surgeon_schedule_copy[sec2][d1][t1 + b] = p1;
    for b in range(dur2):
        secundarios_copy[start2 + b] = sec1;
        surgeon_schedule_copy[sec1][d2][t2 + b] = p2;
        surgeon_schedule_copy[sec2][d2][t2 + b] = -1;
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def MoverPaciente_bloque(sol, OT, nSlot, nDays, AOR=None, fichas=None, dictCosts=None, SP=None, hablar=False):
    """
    Attempt to move the patient by +1/-1 block in the same day,
    checking that new main surgeon cost is feasible in fichas if day changes (though day might not change here),
    or if the main surgeon changes. If main surgeon remains the same, just do the move.
    """
    pac, pri, sec = sol[0].copy(), sol[1].copy(), sol[2].copy()
    feasible_moves = []
    scheduled = [i for i, blk in enumerate(pac) if blk != -1]
    if not scheduled:
        return (pac, pri, sec)
    for p in scheduled:
        old_start = pac[p]
        o, d, t = decompress(old_start, nSlot, nDays)
        dur = OT[p]
        main = pri[old_start]
        for mov in [-1, +1]:
            new_t = t + mov
            if new_t < 0 or new_t + dur > nSlot:
                continue
            free_and_valid = True
            for b in range(dur):
                nb = compress(o, d, new_t + b, nSlot, nDays)
                if nb in pri or nb in sec:
                    free_and_valid = False
                    break
                if AOR is not None:
                    if not is_feasible_block(p, o, d, new_t + b, AOR):
                        free_and_valid = False
                        break
            if free_and_valid:
                feasible_moves.append((p, mov))
    if not feasible_moves:
        return (pac, pri, sec)
    p, mov = random.choice(feasible_moves)
    old_start = pac[p]
    o, d, t = decompress(old_start, nSlot, nDays)
    dur = OT[p]
    main = pri[old_start]
    secondary = sec[old_start]
    for b in range(dur):
        del pri[old_start + b]
        del sec[old_start + b]
    new_t = t + mov
    new_start = compress(o, d, new_t, nSlot, nDays)
    if fichas is not None and dictCosts is not None:
        cost = dictCosts.get((main, secondary, new_start), 0)
        if fichas[(main, d)] < cost:
            for b in range(dur):
                pri[old_start + b] = main
                sec[old_start + b] = secondary
            return (pac, pri, sec)
        fichas[(main, d)] -= cost
    for b in range(dur):
        nb = new_start + b
        pri[nb] = main
        sec[nb] = secondary
    pac[p] = new_start
    if hablar:
        print(f"Patient {p} from block {t} to {new_t}.")
    return (pac, pri, sec)

def MoverPaciente_dia(sol, OT, nSlot, nDays, AOR=None, fichas=None, dictCosts=None, SP=None, hablar=False):
    """
    Similar to MoverPaciente_bloque but tries to shift the entire surgery to day+1 or day-1,
    checking fichas for the main surgeon if the day changes.
    """
    pac, pri, sec = sol[0].copy(), sol[1].copy(), sol[2].copy()
    scheduled = [i for i, blk in enumerate(pac) if blk != -1]
    if not scheduled:
        return (pac, pri, sec)
    feasible_moves = []
    for p in scheduled:
        old_start = pac[p]
        o, d, t = decompress(old_start, nSlot, nDays)
        dur = OT[p]
        main = pri[old_start]
        secondary = sec[old_start]
        for mov in [-1, +1]:
            new_d = d + mov
            if new_d < 0 or new_d >= nDays:
                continue
            free_and_valid = True
            for b in range(dur):
                nb = compress(o, new_d, t + b, nSlot, nDays)
                if nb in pri or nb in sec:
                    free_and_valid = False
                    break
                if AOR is not None:
                    if not is_feasible_block(p, o, new_d, t + b, AOR):
                        free_and_valid = False
                        break
            if free_and_valid:
                feasible_moves.append((p, mov))
    if not feasible_moves:
        return (pac, pri, sec)
    p, mov = random.choice(feasible_moves)
    old_start = pac[p]
    o, d, t = decompress(old_start, nSlot, nDays)
    dur = OT[p]
    main = pri[old_start]
    secondary = sec[old_start]
    for b in range(dur):
        del pri[old_start + b]
        del sec[old_start + b]
    new_d = d + mov
    new_start = compress(o, new_d, t, nSlot, nDays)
    if fichas is not None and dictCosts is not None:
        cost = dictCosts.get((main, secondary, new_start), 0)
        if fichas[(main, new_d)] < cost:
            for b in range(dur):
                pri[old_start + b] = main
                sec[old_start + b] = secondary
            return (pac, pri, sec)
        fichas[(main, new_d)] -= cost
        fichas[(main, d)] += dictCosts.get((main, secondary, old_start), 0)
    for b in range(dur):
        nb = new_start + b
        pri[nb] = main
        sec[nb] = secondary
    pac[p] = new_start
    if hablar:
        print(f"Patient {p} from day {d} to {new_d}.")
    return (pac, pri, sec)

def EliminarPaciente(solucion, surgeon, second, OT, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    #sol = (solucion[0][0].copy(), solucion[0][1].copy(), solucion[0][2].copy());
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    #pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);
    scheduled = [i for i, blk in enumerate(pacientes_copy) if blk != -1];
    if len(scheduled) < 1:
        print("[EliminarPaciente] No hay pacientes para eliminar.") if hablar else None;
        return solucion
    
    p = random.choice(scheduled);
    o, d, t = decompress(pacientes_copy[p], nSlot, nDays);
    dur = OT[p];
    start = pacientes_copy[p];
    main = primarios_copy[start];
    sec = secundarios_copy[start];
    for b in range(dur):
        del primarios_copy[start + b];
        surgeon_schedule_copy[main][d][t + b] = -1;
        del secundarios_copy[start + b];
        surgeon_schedule_copy[sec][d][t + b] = -1;
        or_schedule_copy[o][d][t + b] = -1;
    for d_aux in range(d, nDays):
        fichas_copy[(main, d_aux)] += dictCosts[(main, sec, start)];
    pacientes_copy[p] = -1;
    print(f"[EliminarPaciente] Paciente {p} eliminado del bloque {start}.") if hablar else None;
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def AgregarPaciente(solucion, surgeon, second, OT, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    sol = (solucion[0][0].copy(), solucion[0][1].copy(), solucion[0][2].copy());
    surgeon_schedule = copy.deepcopy(solucion[1]);
    or_schedule = copy.deepcopy(solucion[2]);
    fichas = copy.deepcopy(solucion[3]);
    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    unscheduled = [i for i, blk in enumerate(pacientes) if blk == -1];
    if len(unscheduled) < 1:
        print("[AgregarPaciente] No hay pacientes para agregar.") if hablar else None;
        return solucion
    
    all_start_blocks = []
    for quir in room:
        for d_ in range(nDays):
            for t_ in range(nSlot):
                all_start_blocks.append(quir * nSlot * nDays + d_ * nSlot + t_)
    random.shuffle(all_start_blocks)
    assigned = False
    for start_block in all_start_blocks:
        o_asign = start_block // (nSlot * nDays)
        tmp = start_block % (nSlot * nDays)
        d_asign = tmp // nSlot
        t_asign = tmp % nSlot
        feasible_candidates = []
        for p in unscheduled:
            dur = OT[p]
            if t_asign + dur > nSlot:
                continue
            posible = True
            for b in range(dur):
                blk = start_block + b
                if AOR[p][o_asign][t_asign + b][d_asign % 5] != 1:
                    posible = False
                    break
                if blk in primarios or blk in secundarios:
                    posible = False
                    break
            if posible:
                prioridad = I[(p, d_asign)]
                feasible_candidates.append((p, prioridad))
        if feasible_candidates:
            feasible_candidates.sort(key=lambda x: x[1], reverse=True)
            best_p, best_priority = feasible_candidates[0]
            compatible_main_surgeons = [s for s in surgeon if SP[best_p][s] == 1]
            if not compatible_main_surgeons:
                continue
            main_s = None
            for cand_s in compatible_main_surgeons:
                cost = dictCosts.get((cand_s, None, start_block), 10)
                if fichas[(cand_s, d_asign)] >= cost:
                    main_s = cand_s
                    fichas[(cand_s, d_asign)] -= cost
                    break
            if main_s is None:
                continue
            sec_s = random.choice(second)
            pacientes[best_p] = start_block
            dur = OT[best_p]
            for b in range(dur):
                blk = start_block + b
                primarios[blk] = main_s
                secundarios[blk] = sec_s
            assigned = True
            if hablar:
                print(f"Asig p={best_p} prio={best_priority}, OR={o_asign}, dia={d_asign}, slot={t_asign}, dur={dur}, main={main_s}, sec={sec_s}")
            break
    if not assigned and hablar:
        print("No se pudo asignar ningún paciente con fichas y prioridad.")
    return (pacientes, primarios, secundarios)

def AgregarPaciente_1(solucion, surgeon, second, OT, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    #sol = (solucion[0][0].copy(), solucion[0][1].copy(), solucion[0][2].copy());
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    #pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);
    unscheduled = [i for i, blk in enumerate(pacientes_copy) if blk == -1];
    if len(unscheduled) < 1:
        print("[AgregarPaciente_1] No hay pacientes para agregar.") if hablar else None;
        return solucion

    all_start_blocks = [];
    for o in range(len(or_schedule_copy)):
        for d in range(nDays):
            for t  in range(nSlot):
                if or_schedule_copy[o][d][t] == -1:
                    all_start_blocks.append(compress(o, d, t, nSlot, nDays));
    random.shuffle(all_start_blocks);
    if len(all_start_blocks) == 0:
        print("[AgregarPaciente_1] No hay bloques disponibles.") if hablar else None;
        return solucion
    
    assigned = False;
    for start_block in all_start_blocks:
        o, d, t = decompress(start_block, nSlot, nDays);
        feasible_patients = [];
        for p in unscheduled:
            dur = OT[p];
            if (t + dur >= nSlot) or (t < nSlot//2 and t + dur >= nSlot//2):
                continue
            posible = True;
            if AOR[p][o][t][d % 5] != 1:
                posible = False;
                break
            for b in range(dur):
                if or_schedule_copy[o][d][t + b] != -1:
                    posible = False;
                    break;
            if posible:
                feasible_patients.append(p);
        
        if len(feasible_patients) > 0:
            chosen_p = random.choice(feasible_patients);
            dur = OT[chosen_p];
            comp_mains = [s for s in surgeon if SP[chosen_p][s] == 1 and all(surgeon_schedule_copy[s][d][t + b] == -1 for b in range(dur))];
            if len(comp_mains) == 0:
                continue
            main_s = None;
            for cm in comp_mains:
                comp_second = [a for a in second if a != cm and all(surgeon_schedule_copy[a][d][t + b] == -1 for b in range(dur))];
                if len(comp_second) == 0:
                    continue;
                second_s = random.choice(comp_second);
                c = dictCosts[(cm, second_s, start_block)];
                if all(fichas_copy[(cm, d_aux)] >= c for d_aux in range(d, nDays)):
                    main_s = cm;
                    break
            if main_s is None:
                continue

            pacientes_copy[chosen_p] = start_block;
            for b in range(dur):
                blk = start_block + b;
                primarios_copy[blk] = main_s;
                surgeon_schedule_copy[main_s][d][t + b] = chosen_p;
                secundarios_copy[blk] = second_s;
                surgeon_schedule_copy[second_s][d][t + b] = chosen_p;
                or_schedule_copy[o][d][t + b] = chosen_p;
                for d_aux in range(d, nDays):
                    fichas_copy[(main_s, d_aux)] -= c;
            assigned = True;
            print(f"[AgregarPaciente_1] p={chosen_p} OR={o}, d={d}, slot={t}, main={main_s}, sec={second_s}") if hablar else None;
            break
    if not assigned and hablar:
        print("[AgregarPaciente_1] No se pudo asignar ningún paciente.");
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def AgregarPaciente_2(solucion, surgeon, second, OT, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);

    unscheduled = [p for p, blk in enumerate(pacientes_copy) if blk == -1]
    if not unscheduled:
        if hablar:
            print("[AgregarPaciente_2] No hay pacientes para agregar.")
        return solucion

    chosen_p = random.choice(unscheduled);
    dur = OT[chosen_p];

    all_start_blocks = []
    for o in range(len(or_schedule_copy)):
        for d_ in range(nDays):
            for t_ in range(nSlot):
                if or_schedule_copy[o][d_][t_] == -1:
                    all_start_blocks.append(compress(o, d_, t_, nSlot, nDays));

    random.shuffle(all_start_blocks);
    assigned = False;

    for start_block in all_start_blocks:
        o, d_, t_ = decompress(start_block, nSlot, nDays);
        if t_ + dur > nSlot:
            continue;
        if t_ < nSlot//2 and (t_ + dur) > nSlot // 2:
            continue;
        posible = True
        for b in range(dur):
            if or_schedule_copy[o][d_][t_ + b] != -1:
                posible = False;
                break
        if AOR[chosen_p][o][t_][d_ % 5] != 1:
            posible = False;
        if not posible:
            continue;

        comp_mains = [s for s in surgeon if SP[chosen_p][s] == 1 and all(surgeon_schedule_copy[s][d_][t_ + b] == -1 for b in range(dur))]
        if not comp_mains:
            continue;

        main_s = None;
        second_s = None;
        for cm in comp_mains:
            comp_second = [a for a in second if a != cm and all(surgeon_schedule_copy[a][d_][t_ + b] == -1 for b in range(dur))];
            if not comp_second:
                continue;
            cand_second = random.choice(comp_second)
            cost = dictCosts[(cm, cand_second, start_block)];
            if all(fichas_copy[(cm, day_idx)] >= cost for day_idx in range(d_, nDays)):
                main_s = cm;
                second_s = cand_second;
                break

        if main_s is None:
            continue;

        pacientes_copy[chosen_p] = start_block;
        for b in range(dur):
            blk = start_block + b;
            primarios_copy[blk] = main_s;
            secundarios_copy[blk] = second_s;

            surgeon_schedule_copy[main_s][d_][t_ + b] = chosen_p;
            surgeon_schedule_copy[second_s][d_][t_ + b] = chosen_p;
            or_schedule_copy[o][d_][t_ + b] = chosen_p;

        for day_idx in range(d_, nDays):
            fichas_copy[(main_s, day_idx)] -= cost;

        assigned = True;
        print(f"[AgregarPaciente_2] p={chosen_p} OR={o}, d={d_}, slot={t_}, main={main_s}, sec={second_s}") if hablar else None;
        break

    if not assigned and hablar:
        print(f"[AgregarPaciente_2] No se encontró bloque factible para p={chosen_p}.")
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)