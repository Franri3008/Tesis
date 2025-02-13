import random

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

def CambiarPrimarios(sol, OT, SP=None, fichas=None, dictCosts=None, nSlot=None, nDays=None, hablar=False):
    """
    Swap the main surgeons of two different scheduled patients, only if:
    1. Each new main surgeon is allowed for that patient (SP).
    2. There's enough fichas on the day for each new assignment.
    3. The swap is otherwise feasible.
    """
    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy()
    scheduled = [i for i, blk in enumerate(pacientes) if blk != -1]
    if len(scheduled) < 2:
        if hablar:
            print("[CambiarPrimarios] No hay suficientes pacientes para el cambio.")
        return (pacientes, primarios, secundarios)

    valid_pairs = []
    for i in range(len(scheduled)):
        for j in range(i + 1, len(scheduled)):
            p1 = scheduled[i]
            p2 = scheduled[j]
            start1 = pacientes[p1]
            start2 = pacientes[p2]
            cir1 = primarios[start1]
            cir2 = primarios[start2]
            sec1 = secundarios[start1]
            sec2 = secundarios[start2]
            can_swap = True
            if SP is not None:
                if SP[p1][cir2] != 1 or SP[p2][cir1] != 1 or cir1 == sec2 or cir2 == sec1:
                    can_swap = False
            if can_swap:
                valid_pairs.append((p1, p2))

    if not valid_pairs:
        return (pacientes, primarios, secundarios)

    p1, p2 = random.choice(valid_pairs)
    s1 = pacientes[p1]
    s2 = pacientes[p2]
    old_main1 = primarios[s1]
    old_main2 = primarios[s2]
    sec1 = secundarios[s1]
    sec2 = secundarios[s2]
    dur1 = OT[p1]
    dur2 = OT[p2]
    o1, d1, t1 = decompress(s1, nSlot, nDays)
    o2, d2, t2 = decompress(s2, nSlot, nDays)
    new_main_for_p1 = old_main2
    new_main_for_p2 = old_main1
    if dictCosts is not None:
        c1 = dictCosts.get((new_main_for_p1, sec1, s1), 999999)
        c2 = dictCosts.get((new_main_for_p2, sec2, s2), 999999)
    else:
        c1 = 0
        c2 = 0
    enough_f1 = True
    enough_f2 = True
    if fichas is not None:
        if fichas[(new_main_for_p1, d1)] < c1:
            enough_f1 = False
        if fichas[(new_main_for_p2, d2)] < c2:
            enough_f2 = False
    if not (enough_f1 and enough_f2):
        if hablar:
            print("[CambiarPrimarios] No hay fichas suficientes para hacer el swap.")
        return (pacientes, primarios, secundarios)
    if fichas is not None:
        fichas[(new_main_for_p1, d1)] -= c1
        fichas[(new_main_for_p2, d2)] -= c2
    if hablar:
        print(f"Cambiando cirujanos p={p1}({old_main1}) <-> p={p2}({old_main2}).")
    for b in range(dur1):
        primarios[s1 + b] = new_main_for_p1
    for b in range(dur2):
        primarios[s2 + b] = new_main_for_p2
    return (pacientes, primarios, secundarios)

def CambiarSecundarios(sol, OT, SP=None, hablar=False):
    """
    Swap the secondary surgeons of two different scheduled patients,
    but only if it is feasible.
    Otherwise do nothing.
    """
    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy()
    scheduled = [i for i, blk in enumerate(pacientes) if blk != -1]
    if len(scheduled) < 2:
        return (pacientes, primarios, secundarios)
    valid_pairs = []
    for i in range(len(scheduled)):
        for j in range(i + 1, len(scheduled)):
            p1 = scheduled[i]
            p2 = scheduled[j]
            t1 = pacientes[p1]
            t2 = pacientes[p2]
            sec1 = secundarios[t1]
            sec2 = secundarios[t2]
            cir1 = primarios[t1]
            cir2 = primarios[t2]
            can_swap = True
            if cir1 == sec2 or cir2 == sec1:
                can_swap = False
            if can_swap:
                valid_pairs.append((p1, p2))
    if not valid_pairs:
        return (pacientes, primarios, secundarios)
    p1, p2 = random.choice(valid_pairs)
    t1 = pacientes[p1]
    t2 = pacientes[p2]
    sec1 = secundarios[t1]
    sec2 = secundarios[t2]
    dur1 = OT[p1]
    dur2 = OT[p2]
    if hablar:
        print(f"Swapping secondary between p={p1} and p={p2}.")
    for b in range(dur1):
        secundarios[t1 + b] = sec2
    for b in range(dur2):
        secundarios[t2 + b] = sec1
    return (pacientes, primarios, secundarios)

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

def EliminarPaciente(sol, OT, hablar=False):
    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy()
    scheduled = [i for i, blk in enumerate(pacientes) if blk != -1]
    if not scheduled:
        if hablar:
            print("[EliminarPaciente] No hay pacientes asignados.")
        return (pacientes, primarios, secundarios)
    p = random.choice(scheduled)
    dur = OT[p]
    start = pacientes[p]
    for b in range(dur):
        if start + b in primarios:
            del primarios[start + b]
        if start + b in secundarios:
            del secundarios[start + b]
    pacientes[p] = -1
    if hablar:
        print(f"[EliminarPaciente] Paciente {p} eliminado del bloque {start}.")
    return (pacientes, primarios, secundarios)

def AgregarPaciente_old(sol, AOR, OT, nSlot, nDays, room, slot, day, surgeon, second, hablar=False):
    """
    Already had a feasibility check with AOR, but you can add any additional
    checks needed for the "specialty" or surgeon constraints.
    """
    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy()
    candidatos = [i for i, x in enumerate(pacientes) if x == -1]
    if not candidatos:
        if hablar:
            print("No hay pacientes para agregar.")
        return (pacientes, primarios, secundarios)
    p = random.choice(candidatos)
    dur = OT[p]
    espacios_disponibles = []
    for quir in room:
        for d_ in day:
            for t_ in slot:
                if t_ + dur - 1 >= len(slot):
                    continue
                posible = True
                for b in range(dur):
                    if AOR[p][quir][t_ + b][d_ % 5] != 1:
                        posible = False
                        break
                    blk = quir * nSlot * nDays + d_ * nSlot + (t_ + b)
                    if blk in primarios or blk in secundarios:
                        posible = False
                        break
                if posible:
                    espacios_disponibles.append(quir * nSlot * nDays + d_ * nSlot + t_)
    if not espacios_disponibles:
        if hablar:
            print("No hay espacio para asignar.")
        return (pacientes, primarios, secundarios)
    block_inicial = random.choice(espacios_disponibles)
    o_asign = block_inicial // (nSlot * nDays)
    temp = block_inicial % (nSlot * nDays)
    d_asign = temp // nSlot
    t_asign = temp % nSlot
    if not surgeon or not second:
        if hablar:
            print("No hay cirujanos disponibles en las listas.")
        return (pacientes, primarios, secundarios)
    pacientes[p] = block_inicial
    for b in range(dur):
        blk = block_inicial + b
        cir_primario = random.choice(surgeon)
        cir_secundario = random.choice(second)
        primarios[blk] = cir_primario
        secundarios[blk] = cir_secundario
    if hablar:
        print(f"Paciente {p} asignado. Quirófano={o_asign}, Día={d_asign}, Slot={t_asign}, Duración={dur}.")
    return (pacientes, primarios, secundarios)

def AgregarPaciente(
    sol,
    AOR,
    OT,
    nSlot,
    nDays,
    room,
    slot,
    day,
    surgeon,
    second,
    SP,
    I,
    fichas,
    dictCosts,
    hablar=False
):
    """
    Improved version of AgregarPaciente:
    1) Collect all potential (OR, day, slot) start blocks.
    2) Randomly iterate over them.
    3) For each block, find unscheduled patients that can fit.
    4) Among those, pick the highest-priority patient using I[(p, d_asign)].
    5) Select a main surgeon who is compatible (SP[p][s] == 1) and has enough fichas for day d_asign.
    6) Randomly pick a secondary surgeon from 'second'.
    7) Assign and stop.
    """
    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy()
    unscheduled_patients = [p for p, blk in enumerate(pacientes) if blk == -1]
    if not unscheduled_patients:
        if hablar:
            print("No hay pacientes sin asignar.")
        return (pacientes, primarios, secundarios)
    all_start_blocks = []
    for quir in room:
        for d_ in day:
            for t_ in slot:
                all_start_blocks.append(quir * nSlot * nDays + d_ * nSlot + t_)
    random.shuffle(all_start_blocks)
    assigned = False
    for start_block in all_start_blocks:
        o_asign = start_block // (nSlot * nDays)
        tmp = start_block % (nSlot * nDays)
        d_asign = tmp // nSlot
        t_asign = tmp % nSlot
        feasible_candidates = []
        for p in unscheduled_patients:
            dur = OT[p]
            if t_asign + dur > len(slot):
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

def AgregarPaciente_1(
    sol,
    AOR,
    OT,
    nSlot,
    nDays,
    room,
    slot,
    day,
    surgeon,
    second,
    SP,
    fichas,
    dictCosts,
    hablar=False
):
    """
    AgregarPaciente_1 (Block-First):
    1) Collect all (OR, day, slot) blocks as integers.
    2) Shuffle them.
    3) For each block, gather unscheduled patients that can run there.
    4) Randomly pick one feasible patient, check fichas if we change main surgeon, assign and stop.
    """
    pac, pri, sec = sol[0].copy(), sol[1].copy(), sol[2].copy()
    unscheduled_patients = [p for p, blk in enumerate(pac) if blk == -1]
    if not unscheduled_patients:
        if hablar:
            print("No hay pacientes sin asignar (AgregarPaciente_1).")
        return (pac, pri, sec)
    all_start_blocks = []
    for quir in room:
        for d_ in day:
            for t_ in slot:
                all_start_blocks.append(compress(quir, d_, t_, nSlot, nDays))
    random.shuffle(all_start_blocks)
    assigned = False
    for start_block in all_start_blocks:
        o_asign = start_block // (nSlot * nDays)
        tmp = start_block % (nSlot * nDays)
        d_asign = tmp // nSlot
        t_asign = tmp % nSlot
        feasible_patients = []
        for p in unscheduled_patients:
            dur = OT[p]
            if t_asign + dur > len(slot):
                continue
            posible = True
            for b in range(dur):
                blk = start_block + b
                if AOR[p][o_asign][t_asign + b][d_asign % 5] != 1:
                    posible = False
                    break
                if blk in pri or blk in sec:
                    posible = False
                    break
            if posible:
                feasible_patients.append(p)
        if feasible_patients:
            chosen_p = random.choice(feasible_patients)
            dur = OT[chosen_p]
            comp_mains = [s for s in surgeon if SP[chosen_p][s] == 1]
            if not comp_mains:
                continue
            main_s = None
            for cm in comp_mains:
                c = dictCosts.get((cm, None, start_block), 10)
                if fichas[(cm, d_asign)] >= c:
                    main_s = cm
                    fichas[(cm, d_asign)] -= c
                    break
            if main_s is None:
                continue
            sec_s = random.choice(second)
            pac[chosen_p] = start_block
            for b in range(dur):
                blk = start_block + b
                pri[blk] = main_s
                sec[blk] = sec_s
            assigned = True
            if hablar:
                print(f"[AgregarPaciente_1] p={chosen_p} OR={o_asign}, d={d_asign}, slot={t_asign}, main={main_s}, sec={sec_s}")
            break
    if not assigned and hablar:
        print("No se pudo asignar ningún paciente (AgregarPaciente_1) con fichas.")
    return (pac, pri, sec)

def AgregarPaciente_2(
    sol,
    AOR,
    OT,
    nSlot,
    nDays,
    room,
    slot,
    day,
    surgeon,
    second,
    SP,
    fichas,
    dictCosts,
    hablar=False
):
    """
    AgregarPaciente_2 (Patient-First):
    1) Gather unscheduled patients.
    2) Randomly pick one.
    3) Collect all (OR, day, slot) blocks.
    4) Check feasibility for that patient, pick main with enough fichas, pick sec at random.
    5) Stop if assigned.
    """
    pac, pri, sec = sol[0].copy(), sol[1].copy(), sol[2].copy()
    unscheduled_patients = [p for p, blk in enumerate(pac) if blk == -1]
    if not unscheduled_patients:
        if hablar:
            print("No hay pacientes sin asignar (AgregarPaciente_2).")
        return (pac, pri, sec)
    chosen_p = random.choice(unscheduled_patients)
    dur = OT[chosen_p]
    all_start_blocks = []
    for quir in room:
        for d_ in day:
            for t_ in slot:
                all_start_blocks.append(compress(quir, d_, t_, nSlot, nDays))
    random.shuffle(all_start_blocks)
    assigned = False
    for start_block in all_start_blocks:
        o_asign = start_block // (nSlot * nDays)
        tmp = start_block % (nSlot * nDays)
        d_asign = tmp // nSlot
        t_asign = tmp % nSlot
        if t_asign + dur > len(slot):
            continue
        posible = True
        for b in range(dur):
            blk = start_block + b
            if AOR[chosen_p][o_asign][t_asign + b][d_asign % 5] != 1:
                posible = False
                break
            if blk in pri or blk in sec:
                posible = False
                break
        if posible:
            cmains = [s for s in surgeon if SP[chosen_p][s] == 1]
            if not cmains:
                continue
            main_s = None
            for cma in cmains:
                c = dictCosts.get((cma, None, start_block), 10)
                if fichas[(cma, d_asign)] >= c:
                    main_s = cma
                    fichas[(cma, d_asign)] -= c
                    break
            if main_s is None:
                continue
            sec_s = random.choice(second)
            pac[chosen_p] = start_block
            for b in range(dur):
                blk = start_block + b
                pri[blk] = main_s
                sec[blk] = sec_s
            assigned = True
            if hablar:
                print(f"[AgregarPaciente_2] p={chosen_p} OR={o_asign}, dia={d_asign}, slot={t_asign}, main={main_s}, sec={sec_s}")
            break
    if not assigned and hablar:
        if hablar:
            print(f"No se encontró bloque factible para p={chosen_p} (AgregarPaciente_2).")
    return (pac, pri, sec)