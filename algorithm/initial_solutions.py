import random
import math

def compress(o, d, t, nSlot, nDays):
    return o * nSlot * nDays + d * nSlot + t

def decompress(val, nSlot, nDays):
    o = val // (nSlot * nDays)
    temp = val % (nSlot * nDays)
    d = temp // nSlot
    t = temp % nSlot
    return o, d, t

def normal(surgeon, second, patient, room, day, slot, AOR, I, dictCosts, nFichas, nSlot, SP, COIN, OT, alpha=0.1, VERSION="C", hablar=False):
    all_personnel = set(surgeon).union(second);
    timeUsedMap = { person: set() for person in all_personnel};
    boundary = nSlot//2;

    def encontrar_pacientes_cirujanos(p):
        compatibles = [];
        for s in surgeon:
            if SP[p][s] == 1:
                for a in second:
                    if a != s and COIN[s][a] == 0:
                        compatibles.append((p, s, a, OT[p]));
        return compatibles

    def cirujano_disponible(s, a, o, d, t, duracion):
        for b in range(int(duracion)):
            if (d, t + b) in timeUsedMap[s]:
                return False
            if (d, t + b) in timeUsedMap[a]:
                return False
        return True

    def asignar_paciente(p, s, a, o, d, t, duracion):
        if asignP[p] == -1:
            asignP[p] = compress(o, d, t);
            for b in range(int(duracion)):
                or_schedule[o][d][t + b] = p;
                surgeon_schedule[s][d][t + b] = p;
                surgeon_schedule[a][d][t + b] = p;

                id_block = compress(o, d, t + b);
                dictS[id_block] = s;
                dictA[id_block] = a;
                timeUsedMap[s].add((d, t + b));
                timeUsedMap[a].add((d, t + b));
            asignS[s].add((o, d, t, duracion));
            asignA[a].add((o, d, t, duracion));

    patient_sorted = sorted(patient, key=lambda p: I[(p, 0)], reverse=True);

    asignP = [-1] * len(patient);
    asignS = {s: set() for s in surgeon};
    asignA = {a: set() for a in second};
    dictS  = {};
    dictA  = {};
    fichas = {(s, d): nFichas * (d+1) for s in surgeon for d in day};

    surgeon_schedule = {s: [[-1 for t in slot] for d in day] for s in surgeon};
    or_schedule = {o: [[-1 for t in slot] for d in day] for o in room};

    for p in patient_sorted:
        assigned = False
        duracion_p = OT[p]
        for o in room:
            for d in day:
                for t in range(nSlot - duracion_p + 1):
                    if duracion_p > 1:
                        if t < boundary and (t + duracion_p) > boundary:
                            continue
                    if all(AOR[p][o][t + b][d % 5] == 1 for b in range(duracion_p)):
                        #if es_bloque_disponible(o, d, t, duracion_p):
                        if all(or_schedule[o][d][t + b] == -1 for b in range(duracion_p)):
                            resultados = encontrar_pacientes_cirujanos(p)
                            for (p_res, s, a, dur) in resultados:
                                if cirujano_disponible(s, a, o, d, t, dur):
                                    cost = dictCosts[(s, a, compress(o, d, t))]
                                    if all(fichas[(s, d_aux)] >= cost*(VERSION=="C") for d_aux in range(d, len(day))):
                                        asignar_paciente(p_res, s, a, o, d, t, dur)
                                        for d_aux in range(d, len(day)):
                                            fichas[(s, d_aux)] -= cost;
                                            #print(f"{d_aux}. costo: {cost}, fichas {s} {d}:", fichas[(s,d)])
                                        assigned = True
                                        break
                                if assigned:
                                    break
                if assigned:
                    break
            if assigned:
                break
    for s in surgeon:
        fichas_por_dia = [fichas[(s, d)] for d in day];
    return (asignP, dictS, dictA), surgeon_schedule, or_schedule, fichas

def GRASP(surgeon, second, patient, room, day, slot, AOR, I, dictCosts, nFichas, nSlot, SP, COIN, OT, alpha=0.1, modo=1, VERSION="C", hablar=False):
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be between 0 (exclusive) and 1 (inclusive)");
    all_personnel = set(surgeon).union(second);
    timeUsedMap = {person: set() for person in all_personnel};
    boundary = nSlot // 2;
    def encontrar_pacientes_cirujanos(p):
        compatibles = []
        for s in surgeon:
            if SP[p][s] == 1:
                for a in second:
                    if a != s and COIN[s][a] == 0:
                        compatibles.append((p, s, a, OT[p]));
        return compatibles

    def cirujano_disponible(s, a, o, d, t, duracion):
        for b in range(int(duracion)):
            slot_time = (d, t + b);
            if slot_time in timeUsedMap.get(s, set()):
                return False
            if slot_time in timeUsedMap.get(a, set()):
                return False
        return True

    def asignar_paciente(p, s, a, o, d, t, duracion):
        if asignP[p] == -1:
            asignP[p] = compress(o, d, t, nSlot, len(day));
            for b in range(int(duracion)):
                current_slot = t + b;
                or_schedule[o][d][current_slot] = p;
                surgeon_schedule[s][d][current_slot] = p;
                surgeon_schedule[a][d][current_slot] = p;

                id_block = compress(o, d, current_slot, nSlot, len(day));
                dictS[id_block] = s;
                dictA[id_block] = a;
                timeUsedMap[s].add((d, current_slot));
                timeUsedMap[a].add((d, current_slot));
            asignS[s].add((o, d, t, duracion));
            asignA[a].add((o, d, t, duracion));
            return True
        return False

    asignP = [-1] * len(patient);
    asignS = {s: set() for s in surgeon};
    asignA = {a: set() for a in second};
    dictS = {};
    dictA = {};
    fichas = {(s, d): nFichas * (d + 1) for s in surgeon for d in day};

    surgeon_schedule = {s: [[-1 for t in slot] for d in day] for s in surgeon};
    or_schedule = {o: [[-1 for t in slot] for d in day] for o in room};

    unassigned_patients = set(patient);

    while unassigned_patients:
        candidates = list(unassigned_patients);
        if modo == 1:
            candidates.sort(key=lambda p: I[(p, 0)], reverse=True);
        elif modo == 2:
            candidates.sort(key=lambda p: OT[p], reverse=False);
        else:
            candidates.sort(key=lambda p: I[(p, 0)]/OT[p], reverse=True);
        
        if not candidates:
            break;
        best_score = I[candidates[0], 0] if candidates else 0;
        min_rcl_score = best_score * (1.0 - alpha);
        # max_score = best_score
        # min_score = I.get((candidates[-1], 0), 0) if candidates else 0
        # threshold = max_score - alpha * (max_score - min_score)
        rcl = [p for p in candidates if I[(p, 0)] >= min_rcl_score];
        if not rcl:
             if candidates:
                 rcl = [candidates[0]];
             else:
                 break

        selected_patient = random.choice(rcl);
        p = selected_patient;
        duracion_p = OT[p];
        assigned_this_iteration = False;

        for o in room:
            for d in day:
                for t in range(nSlot - duracion_p + 1):
                    if duracion_p > 1 and t < boundary and (t + duracion_p) > boundary:
                        continue
                    if all(AOR[p][o][t + b][d % 5] == 1 for b in range(duracion_p)) and all(or_schedule[o][d][t + b] == -1 for b in range(duracion_p)):
                        resultados = encontrar_pacientes_cirujanos(p);
                        # resultados.sort(key=lambda res: dictCosts.get((res[1], res[2], compress(o, d, t)), float('inf')));
                        for (p_res, s, a, dur) in resultados:
                            if cirujano_disponible(s, a, o, d, t, dur):
                                affinity_ok = True;
                                if affinity_ok:
                                    cost = dictCosts[(s, a, compress(o, d, t, nSlot, len(day)))];
                                    budget_ok = True
                                    if VERSION == "C":
                                        budget_ok = all(fichas.get((s, d_aux), 0) >= cost for d_aux in range(d, len(day)));
                                    if budget_ok:
                                        if asignar_paciente(p_res, s, a, o, d, t, dur):
                                            if VERSION == "C":
                                                for d_aux in range(d, len(day)):
                                                    fichas[(s, d_aux)] -= cost;
                                            assigned_this_iteration = True;
                                            print(f"Assigned patient {p} to OR {o}, Day {d}, Slot {t} with S{s}, A{a}") if hablar else None;
                                            break;
                        if assigned_this_iteration: break;
                if assigned_this_iteration: break;
            if assigned_this_iteration: break;

        unassigned_patients.remove(selected_patient);
        if not assigned_this_iteration and hablar:
             print(f"Could not assign patient {selected_patient} in this iteration.");

    if hablar:
        assigned_count = sum(1 for x in asignP if x != -1);
        print(f"GRASP construction finished. Assigned {assigned_count}/{len(patient)} patients.");
        print("Fichas restantes (por cirujano/día):")
        for s in surgeon:
            fichas_por_dia = [fichas.get((s, d), 0) for d in day];
            print(f"  Cirujano {s}: {fichas_por_dia}");
        #final_cost = eval_func((asignP, dictS, dictA), VERSION=version);
        #print(f"Costo final: {final_cost}");

    return (asignP, dictS, dictA), surgeon_schedule, or_schedule, fichas