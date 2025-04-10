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
    if AOR[p][o][t][d % 5] == 1:
        return True
    return False

def CambiarPrimarios(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    #sol = (solucion[0][0].copy(), solucion[0][1].copy(), solucion[0][2].copy());
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    #pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);
    scheduled = [i for i, blk in enumerate(pacientes_copy) if blk != -1]
    if len(scheduled) < 2:
        #print("[CambiarPrimarios] No hay suficientes pacientes para el cambio.") if hablar else None;
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
        #print("[CambiarPrimarios] No hay parejas válidas para el swap.") if hablar else None;
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
            fichas_copy[(cir1, d_aux)] -= dictCosts[(cir1, sec2, start2)];
            fichas_copy[(cir2, d_aux)] += dictCosts[(cir2, sec2, start2)];# - dictCosts[(main2, sec1, start1)];

    #print(f"[CambiarPrimarios] Cambiando cirujanos p={p1} ({cir1}) <-> p={p2} ({cir2}).") if hablar else None;
    for b in range(dur1):
        primarios_copy[start1 + b] = cir2;
        surgeon_schedule_copy[cir1][d1][t1 + b] = -1;
        surgeon_schedule_copy[cir2][d1][t1 + b] = p1;
    for b in range(dur2):
        primarios_copy[start2 + b] = cir1;
        surgeon_schedule_copy[cir1][d2][t2 + b] = p2;
        surgeon_schedule_copy[cir2][d2][t2 + b] = -1;
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def CambiarSecundarios(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    #sol = (solucion[0][0].copy(), solucion[0][1].copy(), solucion[0][2].copy());
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    #pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);
    scheduled = [i for i, blk in enumerate(pacientes_copy) if blk != -1]
    if len(scheduled) < 2:
        #print("[CambiarSecundarios] No hay suficientes pacientes para el cambio.") if hablar else None;
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
        #print("[CambiarSecundarios] No hay parejas válidas para el swap.") if hablar else None;
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

    #print(f"[CambiarSecundarios] Cambiando cirujanos p={p1} ({sec1}) <-> p={p2} ({sec2}).") if hablar else None;
    for b in range(dur1):
        secundarios_copy[start1 + b] = sec2;
        surgeon_schedule_copy[sec1][d1][t1 + b] = -1;
        surgeon_schedule_copy[sec2][d1][t1 + b] = p1;
    for b in range(dur2):
        secundarios_copy[start2 + b] = sec1;
        surgeon_schedule_copy[sec1][d2][t2 + b] = p2;
        surgeon_schedule_copy[sec2][d2][t2 + b] = -1;
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def MoverPaciente_bloque(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    pacientes_copy, primarios_copy, secundarios_copy = (copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]));
    scheduled = [p for p, blk in enumerate(pacientes_copy) if blk != -1];
    if not scheduled:
        #print("[MoverPaciente_bloque] No hay pacientes programados.") if hablar else None;
        return solucion

    feasible_moves = [];
    for p in scheduled:
        old_start = pacientes_copy[p];
        o, d, t = decompress(old_start, nSlot, nDays);
        dur = OT[p];
        if old_start not in primarios_copy or old_start not in secundarios_copy:
            continue
        main = primarios_copy[old_start];
        secondary = secundarios_copy[old_start];
        for mov in [-1, 1]:
            new_t = t + mov;
            if new_t < 0 or (new_t + dur) > nSlot:
                continue;
            block_free = True
            for b in range(dur):
                tb = new_t + b
                if or_schedule_copy[o][d][tb] != -1:
                    block_free = False;
                    break;
                if surgeon_schedule_copy[main][d][tb] != -1 or surgeon_schedule_copy[secondary][d][tb] != -1:
                    block_free = False;
                    break;
                if AOR is not None:
                    if not is_feasible_block(p, o, d, tb, AOR):
                        block_free = False;
                        break;
            if not block_free:
                continue;
            new_start = compress(o, d, new_t, nSlot, nDays);
            c = dictCosts[(main, secondary, new_start)];
            if not all(fichas_copy[(main, dd)] + dictCosts[(main, secondary, old_start)]*(mov==1) >= c for dd in range(d, nDays)):
                continue;
            feasible_moves.append((p, mov));
    if not feasible_moves:
        #print("[MoverPaciente_bloque] No hay movimientos factibles.") if hablar else None;
        return solucion

    p, mov = random.choice(feasible_moves);
    old_start = pacientes_copy[p];
    o, d, t = decompress(old_start, nSlot, nDays);
    dur = OT[p];
    main = primarios_copy[old_start];
    secondary = secundarios_copy[old_start];
    new_t = t + mov;
    new_start = compress(o, d, new_t, nSlot, nDays);

    for b in range(dur):
        blk = old_start + b;
        del primarios_copy[blk];
        del secundarios_copy[blk];
        or_schedule_copy[o][d][t + b] = -1;
        surgeon_schedule_copy[main][d][t + b] = -1;
        surgeon_schedule_copy[secondary][d][t + b] = -1;
    for dd in range(d, nDays):
        fichas_copy[(main, dd)] += dictCosts[(main, secondary, old_start)];
        fichas_copy[(main, dd)] -= dictCosts[(main, secondary, new_start)];

    for b in range(dur):
        nb = new_start + b;
        tb = new_t + b;
        primarios_copy[nb] = main;
        secundarios_copy[nb] = secondary;
        or_schedule_copy[o][d][tb] = p;
        surgeon_schedule_copy[main][d][tb] = p;
        surgeon_schedule_copy[secondary][d][tb] = p;
    pacientes_copy[p] = new_start;

    #print(f"[MoverPaciente_bloque] Paciente {p} movido del bloque {t} al bloque {new_t}.") if hablar else None;
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def MoverPaciente_dia(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    pacientes_copy, primarios_copy, secundarios_copy = (copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]));
    scheduled = [p for p, blk in enumerate(pacientes_copy) if blk != -1];
    if len(scheduled) < 2:
        #print("[MoverPaciente_dia] No hay suficientes pacientes para el cambio.") if hablar else None;
        return solucion

    feasible_moves = [];
    for p in scheduled:
        old_start = pacientes_copy[p];
        o, d, t = decompress(old_start, nSlot, nDays);
        dur = OT[p];
        if old_start not in primarios_copy or old_start not in secundarios_copy:
            continue;
        main = primarios_copy[old_start];
        secondary = secundarios_copy[old_start];
        for mov in [-1, 1]:
            nd = d + mov;
            if nd < 0 or nd >= nDays:
                continue;
            if t + dur > nSlot:
                continue;
            block_free = True;
            for b in range(dur):
                tb = t + b;
                if or_schedule_copy[o][nd][tb] != -1:
                    block_free = False;
                    break
                if surgeon_schedule_copy[main][nd][tb] != -1 or surgeon_schedule_copy[secondary][nd][tb] != -1:
                    block_free = False;
                    break
                if not is_feasible_block(p, o, nd, tb, AOR):
                    block_free = False
                    break
            if not block_free:
                continue
            new_start = compress(o, nd, t, nSlot, nDays);
            cost = dictCosts[(main, secondary, new_start)];
            if all(fichas_copy[(main, dd)] + dictCosts[(main, secondary, old_start)]*(mov==1) >= cost for dd in range(nd, nDays)):
                feasible_moves.append((p, mov));

    if not feasible_moves:
        #print("[MoverPaciente_dia] No hay movimientos factibles.") if hablar else None;
        return solucion

    p, mov = random.choice(feasible_moves);
    old_start = pacientes_copy[p];
    o, d, t = decompress(old_start, nSlot, nDays);
    dur = OT[p];
    main = primarios_copy[old_start]; 
    secondary = secundarios_copy[old_start];
    new_d = d + mov;
    new_start = compress(o, new_d, t, nSlot, nDays);
    c = dictCosts[(main, secondary, new_start)];

    for b in range(dur):
        blk = old_start + b;
        del primarios_copy[blk];
        del secundarios_copy[blk];
        or_schedule_copy[o][d][t + b] = -1;
        surgeon_schedule_copy[main][d][t + b] = -1;
        surgeon_schedule_copy[secondary][d][t + b] = -1;
    for dd in range(d, nDays):
        fichas_copy[(main, dd)] += dictCosts[(main, secondary, old_start)];

    for b in range(dur):
        nb = new_start + b;
        tb = t + b;
        primarios_copy[nb] = main;
        secundarios_copy[nb] = secondary;
        or_schedule_copy[o][new_d][tb] = p;
        surgeon_schedule_copy[main][new_d][tb] = p;
        surgeon_schedule_copy[secondary][new_d][tb] = p;
    for dd in range(new_d, nDays):
        fichas_copy[(main, dd)] -= c;
    pacientes_copy[p] = new_start;

    #print(f"[MoverPaciente_dia] Paciente {p} movido desde el día {d} hasta el día {new_d}.") if hablar else None;
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def EliminarPaciente(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    #sol = (solucion[0][0].copy(), solucion[0][1].copy(), solucion[0][2].copy());
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    #pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);
    scheduled = [i for i, blk in enumerate(pacientes_copy) if blk != -1];
    if len(scheduled) < 1:
        #print("[EliminarPaciente] No hay pacientes para eliminar.") if hablar else None;
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
    #print(f"[EliminarPaciente] Paciente {p} eliminado del bloque {start}.") if hablar else None;
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def AgregarPaciente_1(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    #sol = (solucion[0][0].copy(), solucion[0][1].copy(), solucion[0][2].copy());
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
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
            chosen_second_s = None # Variable to store the chosen secondary
            chosen_cost = None     # Variable to store the chosen cost
            for cm in comp_mains:
                comp_second = [a for a in second if a != cm and all(surgeon_schedule_copy[a][d][t + b] == -1 for b in range(dur))];
                if len(comp_second) == 0:
                    continue;
                second_s = random.choice(comp_second);
                c = dictCosts[(cm, second_s, start_block)];
                if all(fichas_copy[(cm, d_aux)] >= c for d_aux in range(d, nDays)):
                    main_s = cm;
                    chosen_second = second_s;
                    chosen_cost = c;
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

def AgregarPaciente_2(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);

    unscheduled = [p for p, blk in enumerate(pacientes_copy) if blk == -1]
    if not unscheduled:
        #print("[AgregarPaciente_2] No hay pacientes para agregar.") if hablar else None
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
        #print(f"[AgregarPaciente_2] p={chosen_p} OR={o}, d={d_}, slot={t_}, main={main_s}, sec={second_s}") if hablar else None;
        break

    #if not assigned and hablar:
        #print(f"[AgregarPaciente_2] No se encontró bloque factible para p={chosen_p}.")
    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)

def DestruirAgregar10(solucion, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays, hablar=False):
    surgeon_schedule_copy = copy.deepcopy(solucion[1]);
    or_schedule_copy = copy.deepcopy(solucion[2]);
    fichas_copy = copy.deepcopy(solucion[3]);
    pacientes_copy, primarios_copy, secundarios_copy = copy.deepcopy(solucion[0][0]), copy.deepcopy(solucion[0][1]), copy.deepcopy(solucion[0][2]);

    num_patients = len(pacientes_copy);
    num_ors = len(or_schedule_copy);

    scheduled_patients = [p for p in range(num_patients) if pacientes_copy[p] != -1];
    removed_patient_ids = set();
    num_removed = 0;

    if scheduled_patients:
        num_to_remove = min(10, len(scheduled_patients))
        patients_to_remove = random.sample(scheduled_patients, num_to_remove)
        removed_patient_ids = set(patients_to_remove)

        if hablar:
            print(f"[DestructAndAdd] Removing patients: {patients_to_remove}")

        for p_remove in patients_to_remove:
            start_blk_old = pacientes_copy[p_remove]
            if start_blk_old == -1:
                continue

            try:
                o_old, d_old, t_old = decompress(start_blk_old, nSlot, nDays)
            except Exception as e:
                 if hablar:
                     print(f"[DestructAndAdd] Error decompressing block {start_blk_old} for patient {p_remove} during removal: {e}. Skipping removal.")
                 removed_patient_ids.discard(p_remove) # Don't count as removed if error
                 continue

            if start_blk_old not in primarios_copy or start_blk_old not in secundarios_copy:
                 if hablar:
                     print(f"[DestructAndAdd] Inconsistent state for patient {p_remove} at block {start_blk_old} during removal. Skipping removal.")
                 removed_patient_ids.discard(p_remove)
                 continue

            s = primarios_copy[start_blk_old]
            a = secundarios_copy[start_blk_old]
            dur = OT[p_remove]

            cost_key = (s, a, start_blk_old)
            if cost_key not in dictCosts:
                 if hablar:
                     print(f"[DestructAndAdd] Cost key {cost_key} not found for patient {p_remove} during removal. Cannot refund fichas. Skipping removal.")
                 removed_patient_ids.discard(p_remove)
                 continue
            cost_old = dictCosts[cost_key]

            for b in range(dur):
                ob_old = start_blk_old + b
                primarios_copy.pop(ob_old, None)
                secundarios_copy.pop(ob_old, None)

                if 0 <= o_old < num_ors and 0 <= d_old < nDays and 0 <= t_old + b < nSlot:
                    or_schedule_copy[o_old][d_old][t_old + b] = -1
                if 0 <= s < len(surgeon_schedule_copy) and 0 <= d_old < nDays and 0 <= t_old + b < nSlot:
                     if surgeon_schedule_copy[s][d_old][t_old + b] == p_remove:
                         surgeon_schedule_copy[s][d_old][t_old + b] = -1
                if 0 <= a < len(surgeon_schedule_copy) and 0 <= d_old < nDays and 0 <= t_old + b < nSlot:
                     if surgeon_schedule_copy[a][d_old][t_old + b] == p_remove:
                         surgeon_schedule_copy[a][d_old][t_old + b] = -1

            for d_aux in range(d_old, nDays):
                fichas_key_refund = (s, d_aux)
                if fichas_key_refund in fichas_copy:
                    fichas_copy[fichas_key_refund] += cost_old
                elif hablar:
                    print(f"[DestructAndAdd] Warning: Fichas key {fichas_key_refund} not found during refund for patient {p_remove}.")

            pacientes_copy[p_remove] = -1
            num_removed += 1

        if hablar:
             print(f"[DestructAndAdd] Successfully removed {num_removed} patients.")

    current_unscheduled = [p for p, blk in enumerate(pacientes_copy) if blk == -1]
    candidates_to_add = [p for p in current_unscheduled if p not in removed_patient_ids]
    num_added = 0

    if candidates_to_add:
        num_to_add = min(10, len(candidates_to_add))
        patients_to_add = random.sample(candidates_to_add, num_to_add)

        if hablar:
            print(f"[DestructAndAdd] Attempting to add patients: {patients_to_add}")

        for p_add in patients_to_add:
            dur = OT[p_add]
            assignment_found_for_p = False
            main_s = None
            chosen_second_s = None
            chosen_cost = None
            chosen_start_block = -1
            chosen_o = -1
            chosen_d = -1
            chosen_t = -1

            potential_starts = []
            for o in range(num_ors):
                for d in range(nDays):
                    for t in range(nSlot):
                        if t + dur <= nSlot and or_schedule_copy[o][d][t] == -1:
                             # Check full duration OR availability
                             if all(or_schedule_copy[o][d][t + b] == -1 for b in range(dur)):
                                 potential_starts.append(compress(o, d, t, nSlot, nDays))
            random.shuffle(potential_starts)

            for start_block in potential_starts:
                o, d, t = decompress(start_block, nSlot, nDays)

                if t < nSlot // 2 and (t + dur) > nSlot // 2:
                    continue
                if AOR[p_add][o][t][d % 5] != 1:
                    continue

                comp_mains = [s for s in surgeon if SP[p_add][s] == 1 and all(surgeon_schedule_copy[s][d][t + b] == -1 for b in range(dur))]
                if not comp_mains:
                    continue

                found_affordable_combo = False
                for cm in comp_mains:
                    comp_second = [a for a in second if a != cm and all(surgeon_schedule_copy[a][d][t + b] == -1 for b in range(dur))]
                    if not comp_second:
                        continue
                    cand_second = random.choice(comp_second)

                    cost_key = (cm, cand_second, start_block)
                    if cost_key not in dictCosts:
                        continue
                    cost_new = dictCosts[cost_key]

                    can_afford = True
                    for d_check in range(d, nDays):
                        fichas_key = (cm, d_check)
                        if fichas_copy.get(fichas_key, -math.inf) < cost_new:
                            can_afford = False
                            break
                    if not can_afford:
                        continue

                    # Found a valid combo for this block
                    main_s = cm
                    chosen_second_s = cand_second
                    chosen_cost = cost_new
                    chosen_start_block = start_block
                    chosen_o = o
                    chosen_d = d
                    chosen_t = t
                    found_affordable_combo = True
                    break # Break cm loop

                if found_affordable_combo:
                    assignment_found_for_p = True
                    break # Break start_block loop

            # If a valid assignment was found for p_add, perform it
            if assignment_found_for_p:
                pacientes_copy[p_add] = chosen_start_block
                for b in range(dur):
                    blk_new = chosen_start_block + b
                    primarios_copy[blk_new] = main_s
                    secundarios_copy[blk_new] = chosen_second_s

                    surgeon_schedule_copy[main_s][chosen_d][chosen_t + b] = p_add
                    surgeon_schedule_copy[chosen_second_s][chosen_d][chosen_t + b] = p_add
                    or_schedule_copy[chosen_o][chosen_d][chosen_t + b] = p_add

                # Deduct fichas
                for d_aux in range(chosen_d, nDays):
                    fichas_key_deduct = (main_s, d_aux)
                    if fichas_key_deduct in fichas_copy:
                        fichas_copy[fichas_key_deduct] -= chosen_cost
                    elif hablar:
                         print(f"[DestructAndAdd] Warning: Fichas key {fichas_key_deduct} not found during deduction for patient {p_add}.")
                num_added += 1
                if hablar:
                    print(f"[DestructAndAdd] Added patient {p_add} to ({chosen_o},{chosen_d},{chosen_t}) with S={main_s}, A={chosen_second_s}")

        if hablar:
            print(f"[DestructAndAdd] Successfully added {num_added} patients.")

    return ((pacientes_copy, primarios_copy, secundarios_copy), surgeon_schedule_copy, or_schedule_copy, fichas_copy)