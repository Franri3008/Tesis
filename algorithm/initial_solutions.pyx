import random;
import math;
import cython;

cdef inline int c_compress(int o, int d, int t, int nSlot, int nDays):
    return o * nSlot * nDays + d * nSlot + t;

cdef inline tuple c_decompress(int val, int nSlot, int nDays):
    cdef int o = val // (nSlot * nDays);
    cdef int temp_val = val % (nSlot * nDays);
    cdef int d_val = temp_val // nSlot;
    cdef int t_val = temp_val % nSlot;
    return o, d_val, t_val;

cdef list _encontrar_pacientes_cirujanos_cdef(int p_val, list surgeon_list, list second_list, object SP_obj, object COIN_obj, object OT_obj):
    cdef list compatibles = [];
    cdef int s_val, a_val;
    for s_val in surgeon_list:
        if SP_obj[p_val][s_val] == 1:
            for a_val in second_list:
                if a_val != s_val and COIN_obj[s_val][a_val] == 0:
                    compatibles.append((p_val, s_val, a_val, OT_obj[p_val]));
    return compatibles;

cdef bint _cirujano_disponible_cdef(int s_val, int a_val, int d_val, int t_val, int duracion_val, dict timeUsedMap_dict):
    cdef int b_val;
    for b_val in range(duracion_val):
        if (d_val, t_val + b_val) in timeUsedMap_dict.get(s_val, set()):
            return False;
        if (d_val, t_val + b_val) in timeUsedMap_dict.get(a_val, set()):
            return False;
    return True;

cdef void _asignar_paciente_cdef(int p_val, int s_val, int a_val, int o_val, int d_val, int t_val, int duracion_val,
                                 list asignP_list, int nSlot_val, int nDays_val,
                                 dict or_schedule_dict, dict surgeon_schedule_dict,
                                 dict dictS_map, dict dictA_map, dict timeUsedMap_dict,
                                 dict asignS_sets, dict asignA_sets):
    cdef int b_val, id_block, current_slot;
    if asignP_list[p_val] == -1:
        asignP_list[p_val] = c_compress(o_val, d_val, t_val, nSlot_val, nDays_val);
        for b_val in range(duracion_val):
            current_slot = t_val + b_val;
            or_schedule_dict[o_val][d_val][current_slot] = p_val;
            surgeon_schedule_dict[s_val][d_val][current_slot] = p_val;
            surgeon_schedule_dict[a_val][d_val][current_slot] = p_val;

            id_block = c_compress(o_val, d_val, current_slot, nSlot_val, nDays_val);
            dictS_map[id_block] = s_val;
            dictA_map[id_block] = a_val;
            timeUsedMap_dict[s_val].add((d_val, current_slot));
            timeUsedMap_dict[a_val].add((d_val, current_slot));
        asignS_sets[s_val].add((o_val, d_val, t_val, duracion_val));
        asignA_sets[a_val].add((o_val, d_val, t_val, duracion_val));

cpdef tuple normal(list surgeon, list second, list patient, list room, list day, list slot,
                   object AOR, object I, dict dictCosts, double nFichas, int nSlot_val,
                   object SP, object COIN, object OT, str VERSION="C", bint hablar=False):

    cdef set all_personnel = set(surgeon).union(second);
    cdef dict timeUsedMap = {person: set() for person in all_personnel};
    cdef int boundary = nSlot_val // 2;
    cdef int p_val, o_val, d_val, t_loop_val, b_loop_val, d_aux_val;
    cdef int s_res, a_res, dur_res;
    cdef int duracion_p_val;
    cdef bint assigned, aor_check_ok, or_schedule_ok, fichas_ok_check;
    cdef double cost_val;
    cdef list patient_sorted_list;
    cdef list temp_sort_list = [];
    cdef tuple p_s_a_dur_item;

    for p_val in patient:
        temp_sort_list.append((I[(p_val, 0)], p_val));
    temp_sort_list.sort(reverse=True); # No key needed, sorts by first element of tuple
    patient_sorted_list = [item[1] for item in temp_sort_list];

    cdef list asignP = [-1] * len(patient);
    cdef dict asignS = {s: set() for s in surgeon};
    cdef dict asignA = {a: set() for a in second};
    cdef dict dictS_map  = {};
    cdef dict dictA_map  = {};
    cdef dict fichas = {(s, d): nFichas * (d + 1.0) for s in surgeon for d in day};

    cdef dict surgeon_schedule = {s: [[-1 for _t in slot] for _d in day] for s in surgeon};
    cdef dict or_schedule = {o: [[-1 for _t in slot] for _d in day] for o in room};
    cdef int nDays_val = len(day);

    for p_val in patient_sorted_list:
        assigned = False;
        duracion_p_val = <int>OT[p_val];
        for o_val in room:
            for d_val in day:
                for t_loop_val in range(nSlot_val - duracion_p_val + 1):
                    if duracion_p_val > 1:
                        if t_loop_val < boundary and (t_loop_val + duracion_p_val) > boundary:
                            continue;
                    aor_check_ok = True;
                    for b_loop_val in range(duracion_p_val):
                        if AOR[p_val][o_val][t_loop_val + b_loop_val][d_val % 5] != 1:
                            aor_check_ok = False;
                            break;
                    if not aor_check_ok:
                        continue;

                    or_schedule_ok = True;
                    for b_loop_val in range(duracion_p_val):
                        if or_schedule[o_val][d_val][t_loop_val + b_loop_val] != -1:
                            or_schedule_ok = False;
                            break;
                    if not or_schedule_ok:
                        continue;

                    resultados = _encontrar_pacientes_cirujanos_cdef(p_val, surgeon, second, SP, COIN, OT);
                    for p_s_a_dur_item in resultados:
                        s_res = p_s_a_dur_item[1];
                        a_res = p_s_a_dur_item[2];
                        dur_res = <int>p_s_a_dur_item[3];

                        if _cirujano_disponible_cdef(s_res, a_res, d_val, t_loop_val, dur_res, timeUsedMap):
                            cost_val = dictCosts[(s_res, a_res, c_compress(o_val, d_val, t_loop_val, nSlot_val, nDays_val))];
                            
                            fichas_ok_check = True;
                            if VERSION == "C":
                                for d_aux_val in range(d_val, nDays_val):
                                    if fichas[(s_res, d_aux_val)] < cost_val:
                                        fichas_ok_check = False;
                                        break;
                            
                            if fichas_ok_check:
                                _asignar_paciente_cdef(p_val, s_res, a_res, o_val, d_val, t_loop_val, dur_res,
                                                       asignP, nSlot_val, nDays_val, or_schedule, surgeon_schedule,
                                                       dictS_map, dictA_map, timeUsedMap, asignS, asignA);
                                if VERSION == "C":
                                    for d_aux_val in range(d_val, nDays_val):
                                        fichas[(s_res, d_aux_val)] -= cost_val;
                                assigned = True;
                                break;
                    if assigned: break;
                if assigned: break;
            if assigned: break;
    
    return (asignP, dictS_map, dictA_map), surgeon_schedule, or_schedule, fichas;


cpdef tuple GRASP(list surgeon, list second, list patient, list room, list day, list slot,
                  object AOR, object I, dict dictCosts, double nFichas, int nSlot_val,
                  object SP, object COIN, object OT, double alpha=0.1, int modo=1,
                  str VERSION="C", bint hablar=False):

    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be between 0 (exclusive) and 1 (inclusive)");

    cdef set all_personnel = set(surgeon).union(second);
    cdef dict timeUsedMap = {person: set() for person in all_personnel};
    cdef int boundary = nSlot_val // 2;
    cdef int p_val, o_val, d_loop_val, t_loop_val, b_loop_val, d_aux_val;
    cdef int s_res, a_res, dur_res;
    cdef int duracion_p_val, selected_patient_val;
    cdef bint assigned_this_iteration, budget_ok, all_aor_ok, all_or_schedule_ok;
    cdef double cost_val, best_score_val, min_rcl_score_val;
    cdef list candidates_list, rcl_list, temp_sort_list;
    cdef tuple p_s_a_dur_item;
    cdef int assigned_count = 0;
    cdef int x_val;
    cdef int s_val_print, d_val_fichas;
    cdef list fichas_s_dia;
    
    cdef list asignP = [-1] * len(patient);
    cdef dict asignS = {s: set() for s in surgeon};
    cdef dict asignA = {a: set() for a in second};
    cdef dict dictS_map = {};
    cdef dict dictA_map = {};
    cdef dict fichas = {(s, d): nFichas * (d + 1.0) for s in surgeon for d in day};

    cdef dict surgeon_schedule = {s: [[-1 for _t in slot] for _d in day] for s in surgeon};
    cdef dict or_schedule = {o: [[-1 for _t in slot] for _d in day] for o in room};
    cdef int nDays_val = len(day);

    cdef set unassigned_patients_set = set(patient);

    while unassigned_patients_set:
        candidates_list = list(unassigned_patients_set);
        temp_sort_list = [];
        if modo == 1:
            for p_val in candidates_list: temp_sort_list.append((I[(p_val, 0)], p_val));
            temp_sort_list.sort(reverse=True); # No key needed
        elif modo == 2:
            for p_val in candidates_list: temp_sort_list.append((OT[p_val], p_val));
            temp_sort_list.sort(reverse=False); # No key needed, sorts ascending
        else:
            for p_val in candidates_list: temp_sort_list.append((I[(p_val, 0)] / OT[p_val] if OT[p_val] != 0 else float('-inf'), p_val));
            temp_sort_list.sort(reverse=True); # No key needed
        
        candidates_list = [item[1] for item in temp_sort_list];
        
        if not candidates_list:
            break;
        
        best_score_val = I[(candidates_list[0], 0)] if candidates_list else 0.0;
        min_rcl_score_val = best_score_val * (1.0 - alpha);
        
        rcl_list = [p_val for p_val in candidates_list if I[(p_val, 0)] >= min_rcl_score_val];
        if not rcl_list:
             if candidates_list:
                 rcl_list = [candidates_list[0]];
             else:
                 break;

        selected_patient_val = random.choice(rcl_list);
        p_val = selected_patient_val;
        duracion_p_val = <int>OT[p_val];
        assigned_this_iteration = False;

        for o_val in room:
            for d_loop_val in day:
                for t_loop_val in range(nSlot_val - duracion_p_val + 1):
                    if duracion_p_val > 1 and t_loop_val < boundary and (t_loop_val + duracion_p_val) > boundary:
                        continue;
                    
                    all_aor_ok = True;
                    for b_loop_val in range(duracion_p_val):
                        if AOR[p_val][o_val][t_loop_val + b_loop_val][d_loop_val % 5] != 1:
                            all_aor_ok = False;
                            break;
                    if not all_aor_ok: continue;

                    all_or_schedule_ok = True;
                    for b_loop_val in range(duracion_p_val):
                        if or_schedule[o_val][d_loop_val][t_loop_val + b_loop_val] != -1:
                            all_or_schedule_ok = False;
                            break;
                    if not all_or_schedule_ok: continue;
                    
                    resultados = _encontrar_pacientes_cirujanos_cdef(p_val, surgeon, second, SP, COIN, OT);

                    for p_s_a_dur_item in resultados:
                        s_res = p_s_a_dur_item[1];
                        a_res = p_s_a_dur_item[2];
                        dur_res = <int>p_s_a_dur_item[3];

                        if _cirujano_disponible_cdef(s_res, a_res, d_loop_val, t_loop_val, dur_res, timeUsedMap):
                            cost_val = dictCosts[(s_res, a_res, c_compress(o_val, d_loop_val, t_loop_val, nSlot_val, nDays_val))];
                            budget_ok = True;
                            if VERSION == "C":
                                for d_aux_val in range(d_loop_val, nDays_val):
                                    if fichas.get((s_res, d_aux_val), 0.0) < cost_val:
                                        budget_ok = False;
                                        break;
                            
                            if budget_ok:
                                _asignar_paciente_cdef(p_val, s_res, a_res, o_val, d_loop_val, t_loop_val, dur_res,
                                                       asignP, nSlot_val, nDays_val, or_schedule, surgeon_schedule,
                                                       dictS_map, dictA_map, timeUsedMap, asignS, asignA);
                                if VERSION == "C":
                                    for d_aux_val in range(d_loop_val, nDays_val):
                                        fichas[(s_res, d_aux_val)] -= cost_val;
                                assigned_this_iteration = True;
                                if hablar: print(f"Assigned patient {p_val} to OR {o_val}, Day {d_loop_val}, Slot {t_loop_val} with S{s_res}, A{a_res}");
                                break;
                    if assigned_this_iteration: break;
                if assigned_this_iteration: break;
            if assigned_this_iteration: break;

        unassigned_patients_set.remove(selected_patient_val);
        if not assigned_this_iteration and hablar:
             print(f"Could not assign patient {selected_patient_val} in this iteration.");

    if hablar:
        assigned_count = 0;
        for x_val in asignP:
            if x_val != -1:
                assigned_count +=1;
        print(f"GRASP construction finished. Assigned {assigned_count}/{len(patient)} patients.");
        print("Fichas restantes (por cirujano/día):");
        for s_val_print in surgeon:
            fichas_s_dia = [];
            for d_val_fichas in day:
                fichas_s_dia.append(fichas.get((s_val_print, d_val_fichas), 0.0));
            print(f"  Cirujano {s_val_print}: {fichas_s_dia}");

    return (asignP, dictS_map, dictA_map), surgeon_schedule, or_schedule, fichas;

cpdef tuple complete_random(list surgeon, list second, list patient, list room, list day, list slot,
                            object AOR, object I, dict dictCosts, double nFichas, int nSlot_val,
                            object SP, object COIN, object OT, str VERSION="C", bint hablar=False):

    cdef set all_personnel = set(surgeon).union(second);
    cdef dict timeUsedMap = {person: set() for person in all_personnel};
    cdef int boundary = nSlot_val // 2;
    cdef int p_val, o_val, d_val, t_loop_val, b_loop_val, d_aux_val;
    cdef int s_res, a_res;
    cdef int duracion_p_val;
    cdef bint assigned_current_patient, aor_check_ok, or_schedule_ok, fichas_ok_check;
    cdef double cost_val;
    cdef list shuffled_patients = list(patient);
    random.shuffle(shuffled_patients);

    cdef list shuffled_rooms, shuffled_days, shuffled_slots, shuffled_personnel_pairs;
    cdef list possible_personnel_for_p;
    cdef tuple p_s_a_dur_item;
    cdef int assigned_count_random = 0;
    cdef int x_val_count;


    cdef list asignP = [-1] * len(patient);
    cdef dict asignS = {s: set() for s in surgeon};
    cdef dict asignA = {a: set() for a in second};
    cdef dict dictS_map  = {};
    cdef dict dictA_map  = {};
    cdef dict fichas = {(s, d): nFichas * (d + 1.0) for s in surgeon for d in day};

    cdef dict surgeon_schedule = {s: [[-1 for _t_s in slot] for _d_s in day] for s in surgeon};
    cdef dict or_schedule = {o: [[-1 for _t_o in slot] for _d_o in day] for o in room};
    cdef int nDays_val = len(day);

    for p_val in shuffled_patients:
        if asignP[p_val] != -1:
            continue;

        assigned_current_patient = False;
        duracion_p_val = <int>OT[p_val];

        shuffled_rooms = list(room);
        random.shuffle(shuffled_rooms);
        
        possible_personnel_for_p = _encontrar_pacientes_cirujanos_cdef(p_val, surgeon, second, SP, COIN, OT);

        for o_val in shuffled_rooms:
            shuffled_days = list(day);
            random.shuffle(shuffled_days);
            for d_val in shuffled_days:
                
                shuffled_slots = [];
                for t_start in range(nSlot_val - duracion_p_val + 1):
                    shuffled_slots.append(t_start);
                random.shuffle(shuffled_slots);

                for t_loop_val in shuffled_slots:
                    if duracion_p_val > 1:
                        if t_loop_val < boundary and (t_loop_val + duracion_p_val) > boundary:
                            continue;
                    
                    aor_check_ok = True;
                    for b_loop_val in range(duracion_p_val):
                        if AOR[p_val][o_val][t_loop_val + b_loop_val][d_val % 5] != 1:
                            aor_check_ok = False;
                            break;
                    if not aor_check_ok:
                        continue;

                    or_schedule_ok = True;
                    for b_loop_val in range(duracion_p_val):
                        if or_schedule[o_val][d_val][t_loop_val + b_loop_val] != -1:
                            or_schedule_ok = False;
                            break;
                    if not or_schedule_ok:
                        continue;
                    
                    shuffled_personnel_pairs = list(possible_personnel_for_p);
                    random.shuffle(shuffled_personnel_pairs);

                    for p_s_a_dur_item in shuffled_personnel_pairs:
                        s_res = p_s_a_dur_item[1];
                        a_res = p_s_a_dur_item[2];

                        if _cirujano_disponible_cdef(s_res, a_res, d_val, t_loop_val, duracion_p_val, timeUsedMap):
                            cost_val = dictCosts[(s_res, a_res, c_compress(o_val, d_val, t_loop_val, nSlot_val, nDays_val))];
                            
                            fichas_ok_check = True;
                            if VERSION == "C":
                                for d_aux_val in range(d_val, nDays_val):
                                    if fichas[(s_res, d_aux_val)] < cost_val:
                                        fichas_ok_check = False;
                                        break;
                            
                            if fichas_ok_check:
                                _asignar_paciente_cdef(p_val, s_res, a_res, o_val, d_val, t_loop_val, duracion_p_val,
                                                       asignP, nSlot_val, nDays_val, or_schedule, surgeon_schedule,
                                                       dictS_map, dictA_map, timeUsedMap, asignS, asignA);
                                if VERSION == "C":
                                    for d_aux_val in range(d_val, nDays_val):
                                        fichas[(s_res, d_aux_val)] -= cost_val;
                                
                                assigned_current_patient = True;
                                if hablar: print(f"[Random] Assigned patient {p_val} to OR {o_val}, Day {d_val}, Slot {t_loop_val} with S{s_res}, A{a_res}");
                                break; 
                    
                    if assigned_current_patient: break; 
                if assigned_current_patient: break; 
            if assigned_current_patient: break; 
        
        if not assigned_current_patient and hablar:
            print(f"[Random] Could not assign patient {p_val}");

    if hablar:
        assigned_count_random = 0;
        for x_val_count in asignP:
            if x_val_count != -1:
                assigned_count_random +=1;
        print(f"[Random] Construction finished. Assigned {assigned_count_random}/{len(patient)} patients.");

    return (asignP, dictS_map, dictA_map), surgeon_schedule, or_schedule, fichas;