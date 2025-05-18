import cython;
cimport numpy
from libc.math cimport fabs # For abs on doubles

cdef inline int c_compress(int o, int d, int t, int nSlot_val, int nDays_val):
    return o * nSlot_val * nDays_val + d * nSlot_val + t;

cdef inline tuple c_decompress(int val, int nSlot_val, int nDays_val):
    cdef int o = val // (nSlot_val * nDays_val);
    cdef int temp_val = val % (nSlot_val * nDays_val);
    cdef int d_val = temp_val // nSlot_val;
    cdef int t_val = temp_val % nSlot_val;
    return o, d_val, t_val;

cdef double c_evalSchedule_helper(list pac_py, dict prim_py, dict sec_py, int or_id_val, bint hablar_val,
                                  list surgeon_py, object OT_obj, object I_obj, object SP_obj, dict dictCosts_obj,
                                  int nDays_val, int nSlot_val, list fichas_lol):
    cdef dict bloques_por_paciente = {};
    cdef double penalizaciones = 0.0;
    cdef double score_or = 0.0;
    cdef int p_idx, o_p, d_p, t_p, s_c, a_c, s_idx_c, b_c, t_actual_c, bloque_horario_c, d_aux_c;
    cdef double duracion_c, prioridad_paciente_c, cost_c;
    cdef list bloques_c_list; # To store list of blocks for a patient
    cdef int i_c;
    cdef bint is_consecutive;
    cdef int idx_loop, surg_loop_val; # For surgeon.index() replacement

    for p_idx in range(len(pac_py)):
        if pac_py[p_idx] != -1: # Check if patient is scheduled
            o_p, d_p, t_p = c_decompress(pac_py[p_idx], nSlot_val, nDays_val);
            if o_p == or_id_val:
                duracion_c = OT_obj[p_idx];
                prioridad_paciente_c = I_obj[(p_idx, d_p)]; # Assumes I_obj supports this tuple indexing
                s_c = prim_py[pac_py[p_idx]];
                a_c = sec_py[pac_py[p_idx]];

                if p_idx not in bloques_por_paciente:
                    bloques_por_paciente[p_idx] = [];
                    score_or += 1000.0 * prioridad_paciente_c;
                    
                    s_idx_c = -1; # Find index of surgeon s_c
                    for idx_loop_inner, surg_loop_val_inner in enumerate(surgeon_py):
                        if surg_loop_val_inner == s_c:
                            s_idx_c = idx_loop_inner;
                            break;
                    # Add error handling if s_idx_c remains -1 (surgeon not found)
                    if s_idx_c == -1:
                        # This indicates an issue, as s_c should be in surgeon_py
                        # For robustness, you might raise an error here.
                        # print(f"Error: Surgeon {s_c} not found in surgeon list for OR {or_id_val}")
                        continue; # Or handle error appropriately

                    cost_c = dictCosts_obj[(s_c, a_c, pac_py[p_idx])];
                    for d_aux_c in range(d_p, nDays_val):
                        fichas_lol[s_idx_c][d_aux_c] -= cost_c;

                # Ensure p_idx is in bloques_por_paciente before appending
                if p_idx in bloques_por_paciente: # Should always be true due to above block
                    for b_c in range(<int>duracion_c):
                        t_actual_c = t_p + b_c;
                        bloque_horario_c = c_compress(o_p, d_p, t_actual_c, nSlot_val, nDays_val);
                        bloques_por_paciente[p_idx].append(bloque_horario_c);
                        if SP_obj[p_idx][s_c] != 1:
                            penalizaciones += 10.0;
                            if hablar_val:
                                print(f"[OR={or_id_val}] Penalización: Cirujano {s_c} no coincide con paciente {p_idx}.");
                        if s_c == a_c:
                            penalizaciones += 10.0;
                            if hablar_val:
                                print(f"[OR={or_id_val}] Penalización: Cirujano principal y asistente son la misma persona (pIdx={p_idx}).");
                else: # Should not happen if logic is correct
                    if hablar_val: print(f"Logic error: p_idx {p_idx} not initialized in bloques_por_paciente for OR {or_id_val}");


    for paciente_id_key, bloques_c_list_val in bloques_por_paciente.items():
        bloques_c_list_val.sort(); 
        duracion_c = OT_obj[paciente_id_key];
        if len(bloques_c_list_val) != <int>duracion_c:
            penalizaciones += 50.0 * len(bloques_c_list_val);
            if hablar_val:
                print(f"[OR={or_id_val}] Pen: Duración incorrecta para pac {paciente_id_key}.");
        
        is_consecutive = True;
        if len(bloques_c_list_val) > 1:
            for i_c in range(len(bloques_c_list_val) - 1):
                if bloques_c_list_val[i_c] + 1 != bloques_c_list_val[i_c+1]:
                    is_consecutive = False;
                    break;
            if not is_consecutive:
                penalizaciones += 100.0 * len(bloques_c_list_val);
                if hablar_val:
                    print(f"[OR={or_id_val}] Pen: Bloques no consecutivos para pac {paciente_id_key}.");
    
    score_or -= 10.0 * penalizaciones;
    return score_or;

cdef inline double c_multiplicador_helper(int day_idx_val, int nDays_val_param):
    if (day_idx_val + 1) == 0: return 0.0; # Avoid division by zero
    return <double>(nDays_val_param // (day_idx_val + 1));

cpdef double EvalAllORs(tuple sol_py, str VERSION="C", bint hablar=False,
                           # Parameters that were global
                           double nFichas_val=0.0, list day_py=None, list surgeon_py=None, list room_py=None,
                           object OT_obj=None, object I_obj=None, dict dictCosts_obj=None,
                           int nDays_val=0, int nSlot_val=0, object SP_obj=None, double bks=1.0):
    
    cdef list fichas_lol; 
    cdef int d_loop, s_loop;
    
    if day_py is None: day_py = [];
    if surgeon_py is None: surgeon_py = [];
    if room_py is None: room_py = [];

    fichas_lol = [];
    for s_loop in range(len(surgeon_py)):
        row = [];
        for d_loop in range(len(day_py)): # Or use nDays_val if day_py length is guaranteed
            row.append(nFichas_val * (d_loop + 1.0));
        fichas_lol.append(row);

    cdef list pacientes_py = sol_py[0];
    cdef dict primarios_py = sol_py[1];
    cdef dict secundarios_py = sol_py[2];

    cdef double puntaje = 0.0;
    cdef int or_id_loop;
    cdef double score_for_or_val;

    for or_id_loop in room_py:
        score_for_or_val = c_evalSchedule_helper(pacientes_py, primarios_py, secundarios_py, or_id_loop, hablar,
                                                 surgeon_py, OT_obj, I_obj, SP_obj, dictCosts_obj,
                                                 nDays_val, nSlot_val, fichas_lol); # Pass fichas_lol
        puntaje += score_for_or_val;
        if hablar:
            print(f"Score parcial OR={or_id_loop}: {score_for_or_val}");

    cdef int s_idx_loop;
    cdef double penalizacion_fichas_val, leftover_fichas_val;
    cdef double current_fichas_val; # For abs

    for s_idx_loop in range(len(surgeon_py)):
        for d_loop in range(nDays_val): 
            current_fichas_val = fichas_lol[s_idx_loop][d_loop];
            if current_fichas_val < 0:
                # Using Python's abs() for simplicity, or use fabs from libc.math
                penalizacion_fichas_val = 100.0 * abs(current_fichas_val);
                puntaje -= penalizacion_fichas_val;
                if hablar:
                    print(f"Penalización por fichas negativas: Cirujano {surgeon_py[s_idx_loop]}, día {d_loop}, fichas={current_fichas_val}.");

    if VERSION == "C":
        for s_idx_loop in range(len(surgeon_py)):
            for d_loop in range(nDays_val):
                leftover_fichas_val = fichas_lol[s_idx_loop][d_loop];
                puntaje -= leftover_fichas_val * c_multiplicador_helper(d_loop, nDays_val);

    if hablar:
        print("Puntaje final (después de restar fichas' sobrantes):", puntaje);
        print("Fichas restantes (por cirujano/día):");
        for s_idx_loop in range(len(surgeon_py)):
            print(f"  Cirujano {surgeon_py[s_idx_loop]}: {fichas_lol[s_idx_loop]}");
            
    return 1.0 - (puntaje / bks)