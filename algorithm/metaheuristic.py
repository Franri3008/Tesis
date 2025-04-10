#!/usr/bin/env python3
# metaheuristic.py

import sys

import sys
import warnings
warnings.filterwarnings("ignore");
import json
import pandas as pd
import random
import numpy as np
import time
import copy
import math
import importlib

import perturbations
importlib.reload(perturbations)
from perturbations import (
    CambiarPrimarios,
    CambiarSecundarios,
    MoverPaciente_bloque,
    MoverPaciente_dia,
    EliminarPaciente,
    AgregarPaciente_1,
    AgregarPaciente_2,
    DestruirAgregar10
)

import localsearches
importlib.reload(localsearches)
from localsearches import (
    MejorarAfinidad_primario,
    MejorarAfinidad_secundario,
    AdelantarDia,
    MejorOR,
    AdelantarTodos,
    CambiarPaciente1,
    CambiarPaciente2,
    CambiarPaciente3
)

import initial_solutions
importlib.reload(initial_solutions)
from initial_solutions import (
    normal,
    GRASP
)

# ------------------------------------------------------------------------------------
# GLOBAL FLAGS OR CONSTANTS
# ------------------------------------------------------------------------------------
testing = False
parametroFichas = 0.11
entrada = "etapa1"
version = "C"
solucion_inicial = True
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------------

def EvalAllORs(sol, VERSION="C"):
    fichas = [[nFichas * (d+1) for d in range(len(day))] for s in surgeon]
    pacientes, primarios, secundarios = sol

    def evalSchedule(pacientes, primarios, secundarios, or_id):
        bloques_por_paciente = {}
        penalizaciones = 0
        score_or = 0

        for p_idx in range(len(pacientes)):
            if pacientes[p_idx] != -1:
                o_p, d_p, t_p = decompress(pacientes[p_idx])
                if o_p == or_id:
                    duracion = OT[p_idx]
                    prioridad_paciente = I[(p_idx, d_p)]
                    s = primarios[pacientes[p_idx]]
                    a = secundarios[pacientes[p_idx]]

                    if p_idx not in bloques_por_paciente:
                        bloques_por_paciente[p_idx] = []
                        score_or += 1000 * prioridad_paciente
                        s_idx = surgeon.index(s)
                        cost = dictCosts[(s, a, pacientes[p_idx])]
                        for d_aux in range(d_p, nDays):
                            fichas[s_idx][d_aux] -= cost

                    for b in range(int(duracion)):
                        t_actual = t_p + b
                        bloque_horario = compress(o_p, d_p, t_actual)
                        bloques_por_paciente[p_idx].append(bloque_horario)
                        if SP[p_idx][s] != 1:
                            penalizaciones += 10
                        if s == a:
                            penalizaciones += 10

        for paciente_id, bloques in bloques_por_paciente.items():
            bloques.sort()
            duracion = OT[paciente_id]
            if len(bloques) != duracion:
                penalizaciones += 50 * len(bloques)
            if not all(bloques[i]+1 == bloques[i+1] for i in range(len(bloques)-1)):
                penalizaciones += 100 * len(bloques)

        score_or -= 10 * penalizaciones
        return score_or

    puntaje = 0
    for or_id in room:
        score_for_or = evalSchedule(pacientes, primarios, secundarios, or_id)
        puntaje += score_for_or

    for s_idx, s in enumerate(surgeon):
        for d_idx in range(nDays):
            if fichas[s_idx][d_idx] < 0:
                penalizacion_fichas = 100 * abs(fichas[s_idx][d_idx])
                puntaje -= penalizacion_fichas

    if VERSION == "C":
        def multiplicador(day_idx):
            return (nDays // (day_idx + 1))
        for s_idx, s in enumerate(surgeon):
            for d_idx in range(nDays):
                leftover_fichas = fichas[s_idx][d_idx]
                puntaje -= leftover_fichas * multiplicador(d_idx)

    return 1 - (puntaje / bks)

def compress(o, d, t):
    return o * nSlot * nDays + d * nSlot + t

def decompress(val):
    o = (val) // (nSlot * nDays);
    temp = (val) % (nSlot * nDays);
    d = temp // nSlot;
    t = temp % nSlot;
    return o, d, t

def WhichExtra(o,t,d,e):
    return int(extras[o][t][d%5][e]);

# ------------------------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------------------------
def load_data_and_config():
    global dfSurgeon, dfSecond, dfRoom, dfType, dfPatient
    global dfdisORA, dfdisORB, dfdisORC, dfdisORD
    global dfdisAffi, dfdisAffiDiario, dfdisAffiBloque, dfdisRank
    global dfExtra, dfRankExtra, extras, num_ext
    global dfPatient
    global patient, surgeon, second, room, day, nSlot, nRooms, nDays, slot, T, nFichas
    global level_affinity, OT, AOR, COIN, Ex, I, SP
    global dictCosts

    #if testing == False:
    #    with open(f"../input/{entrada}.json") as file:
    #        data = json.load(file)
    #else:
    #    with open("../input/inst_test.json") as file:
    #        data = json.load(file)

    #config = data["configurations"]
    dfSurgeon = pd.read_excel("../input/MAIN_SURGEONS.xlsx", sheet_name='surgeon', converters={'n°': int}, index_col=[0])
    dfSecond = pd.read_excel("../input/SECOND_SURGEONS.xlsx", sheet_name='second', converters={'n°': int}, index_col=[0])
    dfRoom = pd.read_excel("../input/ROOMS.xlsx", sheet_name='room', converters={'n°': int}, index_col=[0])
    dfType = pd.read_excel("../input/PROCESS_TYPE.xls", sheet_name='Process Type', converters={'n°': int}, index_col=[0])

    if typePatients == "low":
        dfPatient = pd.read_csv("../input/LowPriority.csv");
    elif typePatients == "high":
        dfPatient = pd.read_csv("../input/HighPriority.csv");
    else:
        dfPatient = pd.read_csv("../input/AllPriority.csv");
    
    dfPatient = dfPatient.iloc[:nPatients];

    extras = []
    def load_OR_dist(sheet_name):
        df_ = pd.read_excel("../input/DIST_OR_EXT.xlsx", sheet_name=sheet_name, converters={'n°':int}, index_col=[0])
        df_ = df_.astype(str).values.tolist()
        ext_ = []
        for i in range(len(df_)):
            aux = df_[i].copy()
            ext_.append(aux)
        for i in range(len(df_[0])):
            part = df_[0][i].split(";")
            df_[0][i] = part[0]
            ext_[0][i] = part[1:]
            part2 = df_[1][i].split(";")
            df_[1][i] = part2[0]
            ext_[1][i] = part2[1:]
        return df_, ext_

    dfdisORA, extraA = load_OR_dist('A')
    dfdisORB, extraB = load_OR_dist('B')
    dfdisORC, extraC = load_OR_dist('C')
    dfdisORD, extraD = load_OR_dist('D')
    extras.append(extraA)
    extras.append(extraB)
    extras.append(extraC)
    extras.append(extraD)

    # Affinity sheets
    dfdisAffi = pd.read_excel("../input/AFFINITY_EXT.xlsx", sheet_name='Hoja1', converters={'n°':int}, index_col=[0])
    dfdisAffiDiario = pd.read_excel("../input/AFFINITY_DIARIO.xlsx", sheet_name='Dias', converters={'n°':int}, index_col=[0])
    dfdisAffiBloque = pd.read_excel("../input/AFFINITY_DIARIO.xlsx", sheet_name='Bloques', converters={'n°':int}, index_col=[0])
    dfdisRank = pd.read_excel("../input/RANKING.xlsx", sheet_name='Hoja1', converters={'n°':int}, index_col=[0])

    num_ext = 2;

    extra = [];
    dfExtra = [];
    for i in range(num_ext):
        aux = pd.read_excel("../input/AFFINITY_EXT.xlsx", sheet_name='Extra'+str(i+1), converters = {'n°':int}, index_col=[0]);
        extra.append(len(aux));
        dfExtra.append(aux);

    # Rankings for extras
    dfRankExtra = []
    for i in range(num_ext):
        aux = pd.read_excel("../input/RANKING.xlsx", sheet_name='Extra'+str(i+1), converters={'n°':int}, index_col=[0])
        dfRankExtra.append(aux)

    patient = [p for p in range(nPatients)];
    surgeon = [s for s in range(nSurgeons)];
    second = [a for a in range(nSurgeons)]
    room = [o for o in range(len(dfRoom))]
    day = [d for d in range(nDays)];
    nSlot = 16;  # Bloques de 30 minutos
    nRooms = len(room);
    slot = [t for t in range(nSlot)]
    T = nSlot//2; # División entre mañana y tarde

    I = np.ones((nPatients, nDays), dtype=int);
    for p in patient:
        for d in day:
            I[(p,d)] = 1 + dfPatient.iloc[p]["espera"] * dfPatient.iloc[p]["edad"] * 0.0001/(d+1);

    # Rankings para extras
    dfRankExtra = [];
    for i in range(num_ext):
        aux = pd.read_excel("../input/RANKING.xlsx", sheet_name='Extra'+str(i+1), converters = {'n°':int}, index_col=[0]);
        extra.append(len(aux));
        dfRankExtra.append(aux);

    # Matriz de coincidencia cirujanos
    COIN = np.zeros((nSurgeons, nSurgeons), dtype=int);
    for s in surgeon:
        for a in second:
            if dfSurgeon.iloc[s][0] == dfSecond.iloc[a][0]:
                COIN[(s,a)] = 1;
            else:
                COIN[(s,a)] = 0;
                

    Ex = [np.ones((nSurgeons, extra[i]), dtype=float) for i in range(num_ext)];
    for i in range(num_ext):
        for s in surgeon:
            for e in range(extra[i]):
                Ex[i][(s,e)] = dfExtra[i].iloc[e][s+1];

    dictCosts = {};

    for s in surgeon:
        for a in second:
            for _ in range(nSlot * nDays * len(room)):
                o, d, t = decompress(_);
                dictCosts[(s, a, _)] = int(dfdisAffi.iloc[a][s+1] + dfdisAffiDiario.iloc[d%5][s+1] + sum(Ex[i][(s,WhichExtra(o,t//T,d,i)-1)] for i in range(num_ext)) + dfdisAffiBloque.iloc[t//T][s+1]);
    
    DISP = np.ones(nPatients);
    
    process=[t for t in range(len(dfType))] # ?
    def busquedaType(especialidad):
        indice = 0;
        for i in range(len(process)):
            if (especialidad == dfType.iloc[i][0]):
                indice = i;
        return indice
    
    # Compatibilidad paciente-cirujano
    SP = np.zeros((nPatients, nSurgeons), dtype=int);
    contador = 0;
    for p in patient:
        for s in surgeon:
            if busquedaType(dfPatient.iloc[p]["especialidad"]) == busquedaType(dfSurgeon.iloc[s][9]):
                SP[p][s] = 1;       

    OR_specialties = {};
    for o in range(len(room)):
        for d in range(5):
            for t in range(2):
                if o == 0:
                    aux = dfdisORA[t][d];
                elif o == 1:
                    aux = dfdisORB[t][d];
                elif o == 2:
                    aux = dfdisORC[t][d];
                elif o == 3:
                    aux = dfdisORD[t][d];
                OR_specialties[(o, d, t)] = aux;

    # Diccionario de paciente
    dic_p = {p: [0, 0, 0, 0, 0] for p in patient};
    for p in patient:
        #dic_p[p][0] = list_patient[p] #paciente y número aleatorio asociado (entre 0 y 1)
        dic_p[p][1] = busquedaType(dfPatient.iloc[p]["especialidad"]) # ID Especialidad
        dic_p[p][2] = dfPatient.iloc[p]["nombre"]; #Nombre del paciente
        dic_p[p][3] = p; # ID
        dic_p[p][4] = dfPatient.iloc[p]["especialidad"]; # Especialidad requerida

    AOR = np.zeros((nPatients, nRooms, nSlot, 5));
    dicOR = {o:[] for o in room};
    j = [];
    z = [];
    ns = 0;
    for o in room:
        if o == 0:
            for d in range(5):
                for e in range(2):
                    if e == 0:                    
                        for t in range(len(slot)//2):

                            j = dfdisORA[e][d%5];
                            j = j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d%5,t,z]
                                ns+=1
                    if e==1:
                        for t in range(int(len(slot)/2),len(slot)):

                            j=dfdisORA[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d%5,t,z]
                                ns+=1
        if o==1:
            for d in range(5):
                for e in range(2):
                    if e==0:                    
                        for t in range(0,int(len(slot)/2)):
                            j=dfdisORB[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
                    if e==1:
                        for t in range(int(len(slot)/2),len(slot)):
                            j=dfdisORB[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
        if o==2:
            for d in range(5):
                for e in range(2):
                    if e==0:                    
                        for t in range(0,int(len(slot)/2)):
                            j=dfdisORC[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
                    if e==1:
                        for t in range(int(len(slot)/2),len(slot)):
                            j=dfdisORC[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
        if o==3:
            for d in range(5):
                for e in range(2):
                    if e==0:                    
                        for t in range(0,int(len(slot)/2)):
                            j=dfdisORD[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
                    if e==1:
                        for t in range(int(len(slot)/2),len(slot)):
                            j=dfdisORD[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1

    p = 0;
    o = 0;
    t = 0;

    for ns in range(len(dic_p)):
        for nP in range(len(dicOR)):
            if str(dic_p[ns][1])==dicOR[nP][3]:
                p = dic_p[ns][3];
                o = dicOR[nP][0];
                t = dicOR[nP][2];
                d = dicOR[nP][1];
                AOR[p][o][t][d%5] = 1;

    level_affinity = min_affinity;
    OT = np.zeros(nPatients, dtype=int);
    for p in patient:
        OT[p] = int(dfPatient.iloc[p]["tipo"]);
    nFichas = int((parametroFichas * 4 * nSlot * len(room) * 2 * 3 )/(len(surgeon)**0.5));


# ------------------------------------------------------------------------------------
# 2. METAHEURISTIC
# ------------------------------------------------------------------------------------

def destruir_OR(solution):
    surgeon_schedule = copy.deepcopy(solution[1]);
    or_schedule = copy.deepcopy(solution[2]);
    fichas = copy.deepcopy(solution[3]);
    pacientes, primarios, secundarios = copy.deepcopy(solution[0][0]), copy.deepcopy(solution[0][1]), copy.deepcopy(solution[0][2]);

    chosen_or = random.choice(room);
    chosen_day = random.choice(day);
    chosen_time = random.choice([0, nSlot//2]);
    for t_aux in range(chosen_time, chosen_time + nSlot//2):
        p = or_schedule[chosen_or][chosen_day][t_aux];
        if p != -1:
            dur = OT[p];
            block = pacientes[p];
            main_s = primarios[pacientes[p]];
            second_s = secundarios[pacientes[p]];
            for b in range(dur):
                or_schedule[chosen_or][chosen_day][t_aux + b] = -1;
                del primarios[block + b];
                del secundarios[block + b];
                surgeon_schedule[main_s][chosen_day][t_aux + b] = -1;
                surgeon_schedule[second_s][chosen_day][t_aux + b] = -1;
            for d_aux in range(chosen_day, nDays):
                fichas[(main_s, d_aux)] += dictCosts[(main_s, second_s, pacientes[p])];
            pacientes[p] = -1;
    return ((pacientes, primarios, secundarios), surgeon_schedule, or_schedule, fichas)

def metaheuristic(inicial, max_iter=50, destruct_type = 1, destruct=200, temp_inicial=500.0, alpha=0.99,
                  prob_CambiarPrimarios=15, prob_CambiarSecundarios=15, prob_MoverPaciente_bloque=20, prob_MoverPaciente_dia=10,
                  prob_EliminarPaciente=20, prob_AgregarPaciente_1=19, prob_AgregarPaciente_2=19, prob_DestruirAgregar10=2,
                  prob_MejorarAfinidad_primario=20, prob_MejorarAfinidad_secundario=20, prob_AdelantarDia=29, 
                  prob_MejorOR=29, prob_AdelantarTodos=2, prob_CambiarPaciente1=10, prob_CambiarPaciente2=10, prob_CambiarPaciente3=10,
                  prob_Pert=1, prob_Busq=1, semilla=258, GRASP_alpha=0.1, elite_size=5, prob_elite=0.3, prob_GRASP1=0.3, prob_GRASP2=0.3, prob_GRASP3=0.4):
    
    random.seed(semilla);
    initial_time = time.time();

    initial_sol = inicial[0];
    surgeon_schedule = inicial[1];
    or_schedule = inicial[2];
    fichas = inicial[3];

    # [counts, improves, prob]
    metadata_pert = {"CambiarPrimarios": [0, 0, prob_CambiarPrimarios], "CambiarSecundarios": [0, 0, prob_CambiarSecundarios],
                    "MoverPaciente_bloque": [0, 0, prob_MoverPaciente_bloque], "MoverPaciente_dia": [0, 0, prob_MoverPaciente_dia],
                    "EliminarPaciente": [0, 0, prob_EliminarPaciente], "AgregarPaciente_1": [0, 0, prob_AgregarPaciente_1], "AgregarPaciente_2": [0, 0, prob_AgregarPaciente_2],
                    "DestruirAgregar10": [0, 0, prob_DestruirAgregar10], "NoOp": [0, 0, 0]};
    metadata_search = {"MejorarAfinidad_primario": [0, 0, prob_MejorarAfinidad_primario], "MejorarAfinidad_secundario": [0, 0, prob_MejorarAfinidad_secundario],
                       "AdelantarDia": [0, 0, prob_AdelantarDia], "MejorOR": [0, 0, prob_MejorOR], "AdelantarTodos": [0, 0, prob_AdelantarTodos], 
                       "CambiarPaciente1": [0, 0, prob_CambiarPaciente1], "CambiarPaciente2": [0, 0, prob_CambiarPaciente2], 
                       "CambiarPaciente3": [0, 0, prob_CambiarPaciente3], "NoOp": [0, 0, 0]};
    
    lista_evaluacion = [];
    lista_iteracion = [];

    def Perturbar(sol):
        pert_probs = [v[2] for v in metadata_pert.values()];
        total_prob = sum(pert_probs);
        x = random.uniform(0, total_prob);
        cumulative = 0;
        for i, p in enumerate(pert_probs):
            cumulative += p
            if x < cumulative:
                perturbation = list(metadata_pert.keys())[i];
                metadata_pert[perturbation][0] += 1;
                funcion = f"{perturbation}(sol, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays)";
                new_sol = eval(funcion);
                return new_sol, perturbation
        return sol, "NoOp"

    def BusquedaLocal(sol):
        search_probs = [v[2] for v in metadata_search.values()];
        total_prob = sum(search_probs);
        x = random.uniform(0, total_prob);
        cumulative = 0;
        for i, p in enumerate(search_probs):
            cumulative += p;
            if x < cumulative:
                localsearch = list(metadata_search.keys())[i];
                metadata_search[localsearch][0] += 1;
                funcion = f"{localsearch}(sol, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays)";
                new_sol = eval(funcion);
                return new_sol, localsearch
        return sol, "NoOp"

    mejores_sols = [((initial_sol[0].copy(), initial_sol[1].copy(), initial_sol[2].copy()), surgeon_schedule.copy(), or_schedule.copy(), fichas.copy())];
    best_solution = ((initial_sol[0].copy(), initial_sol[1].copy(), initial_sol[2].copy()), surgeon_schedule.copy(), or_schedule.copy(), fichas.copy());
    best_sol = (best_solution[0][0].copy(), best_solution[0][1].copy(), best_solution[0][2].copy());
    best_cost = EvalAllORs(best_sol, VERSION="C");
    elite_pool = [(best_cost, copy.deepcopy(best_solution))];
    current_sol = ((best_sol[0].copy(), best_sol[1].copy(), best_sol[2].copy()), surgeon_schedule.copy(), or_schedule.copy(), fichas.copy());
    current_cost = best_cost;
    T = temp_inicial;
    r = 0;
    d_ = 0;

    for i in range(max_iter):
        if random.uniform(0, 1) < prob_Pert:
            new_sol, last_p = Perturbar(current_sol);
        else:
            new_sol, last_p = copy.deepcopy(current_sol), "NoOp";
        if random.uniform(0, 1) < prob_Busq:
            new_sol, last_s = BusquedaLocal(new_sol);        
        else:
            new_sol, last_s = copy.deepcopy(current_sol), "NoOp";
        
        new_cost = EvalAllORs(new_sol[0], VERSION="C");

        #delta = new_cost - current_cost;
        delta = current_cost - new_cost;
        if delta > 0:
            metadata_pert[last_p][1] += 1;
            metadata_search[last_s][1] += 1;
            current_sol = copy.deepcopy(new_sol);
            current_cost = new_cost;

            if new_cost < best_cost:
                current_sol = copy.deepcopy(new_sol);
                current_cost = new_cost;

                best_solution = copy.deepcopy(current_sol);
                best_cost = current_cost;
            
                elite_pool.append((best_cost, copy.deepcopy(best_solution)));
                elite_pool.sort(key=lambda x: x[0], reverse=False);
                elite_pool = elite_pool[:elite_size];
            d_ = 0;
        else:
            prob_aceptacion = math.exp(delta / T)
            if random.random() < prob_aceptacion:
                current_sol = copy.deepcopy(new_sol);
                current_cost = new_cost;
            else:
                d_ += 1;

        T *= alpha;
        if d_ >= destruct and destruct_type != 0:
            #best_sol = reparar(best_sol)
            mejores_sols.append((copy.deepcopy(current_sol)));
            if destruct_type == 1:
                current_sol = destruir_OR(current_sol);
                current_cost = EvalAllORs(current_sol[0], VERSION="C");
            else:
                if random.random() < prob_elite:
                    _, chosen_elite_sol = random.choice(elite_pool);
                    current_sol = copy.deepcopy(chosen_elite_sol);
                    current_cost = EvalAllORs(current_sol[0], VERSION=version);
                else:
                    pick = random.choices([1, 2, 3], weights=[prob_GRASP1, prob_GRASP2, prob_GRASP3])[0];
                    current_sol = GRASP(surgeon, second, patient, room, day, slot, AOR, I, dictCosts, nFichas, nSlot, SP, COIN, OT, alpha=GRASP_alpha, modo=pick, VERSION="C", hablar=False);
                    current_cost = EvalAllORs(current_sol[0], VERSION=version);
                T = temp_inicial;
            d_ = 0;
        current_time = time.time();
        if current_time - initial_time >= 90:
            mejores_sols.append(copy.deepcopy(current_sol));
            break;
    
    mejores_sols.append(best_solution);
    '''
    mejor_costo = float("inf");
    mejor = None
    for m in mejores_sols:
        try:
            val = EvalAllORs(m[0], VERSION="C")
        except Exception as error:
            val = float("inf");
            with open("./errors.txt", "a") as file:
                file.write(f"{error} \n/// Iteración: {semilla} - {mejores_sols.index(m)}/{len(mejores_sols)}\n\n");
        if val < mejor_costo:
            mejor_costo = val
            mejor = m;
    '''
    mejor_costo, mejor = elite_pool[0];
    #mejor = reparar(mejor)
    #num_asignados_antes = sum(1 for p in mejor[0] if p != -1);
    #mejor = final_add_patients(mejor, VERSION=version);
    #mejor = reparar(mejor)
    #num_asignados_despues = sum(1 for p in mejor[0] if p != -1);
    return mejor, (lista_evaluacion, lista_iteracion, metadata_pert, metadata_search)

# ------------------------------------------------------------------------------------
# 3. MAIN
# ------------------------------------------------------------------------------------
def main():
    global typePatients, nPatients, nDays, min_affinity, nSurgeons, nFichas, time_limit, bks
    if len(sys.argv) != 32:
        print("Usage: metaheuristic.py <instanceID> <seed> <randomSeed> <instanceFile> "
              "<destruct> <temp_inicial> <alpha> <prob_CambiarPrimarios> <prob_CambiarSecundarios>"
              "<prob_MoverPaciente_bloque> <prob_MoverPaciente_dia>" 
              "<prob_EliminarPaciente> <prob_AgregarPaciente_1> <prob_AgregarPaciente_2> <prob_DestruirAgregar10>"
              "<prob_MejorarAfinidad_primario> <prob_MejorarAfinidad_secundario>"
              "<prob_AdelantarDia> <prob_MejorOR> <prob_AdelantarTodos>"
              "<prob_CambiarPaciente1> <prob_CambiarPaciente2> <prob_CambiarPaciente3>"
              "<destruct_type> <prob_Busq> <GRASP_alpha> <elite_size> <prob_elite>" \
              "<prob_GRASP1> <prob_GRASP2> <prob_GRASP3>")
        sys.exit(1)

    # Extract all 31 arguments
    instance_id = sys.argv[1]
    seed = sys.argv[2]
    random_seed = sys.argv[3]
    instance_file = sys.argv[4]
    destruct = int(sys.argv[5])
    temp_inicial = float(sys.argv[6])
    alpha = float(sys.argv[7])

    prob_CambiarPrimarios = float(sys.argv[8])
    prob_CambiarSecundarios = float(sys.argv[9])
    prob_MoverPaciente_bloque = float(sys.argv[10])
    prob_MoverPaciente_dia = float(sys.argv[11])
    prob_EliminarPaciente = float(sys.argv[12])
    prob_AgregarPaciente_1 = float(sys.argv[13])
    prob_AgregarPaciente_2 = float(sys.argv[14])
    prob_DestruirAgregar10 = float(sys.argv[15])

    prob_MejorarAfinidad_primario = float(sys.argv[16])
    prob_MejorarAfinidad_secundario = float(sys.argv[17])
    prob_AdelantarDia = float(sys.argv[18])
    prob_MejorOR = float(sys.argv[19])
    prob_AdelantarTodos = float(sys.argv[20])
    prob_CambiarPaciente1 = float(sys.argv[21])
    prob_CambiarPaciente2 = float(sys.argv[22])
    prob_CambiarPaciente3 = float(sys.argv[23])

    destruct_type = int(sys.argv[24])
    prob_Pert = 1;
    prob_Busq = float(sys.argv[25])
    GRASP_alpha = float(sys.argv[26])
    elite_size = int(sys.argv[27])
    prob_elite = float(sys.argv[28])

    prob_GRASP1 = float(sys.argv[29])
    prob_GRASP2 = float(sys.argv[30])
    prob_GRASP3 = float(sys.argv[31])

    random.seed(seed);
    max_iter = 125000;

    with open(instance_file, 'r') as f:
        data = json.load(f);

    typePatients = data["patients"];
    nPatients = int(data["n_patients"]);
    nDays = int(data["days"]);
    min_affinity = int(data["min_affinity"]);
    nSurgeons = int(data["surgeons"]);
    nFichas = int(data["fichas"]);
    time_limit = int(data["time_limit"]);
    bks = int(data["bks"]);

    load_data_and_config();
    #inicial = normal(surgeon, second, patient, room, day, slot, AOR, I, dictCosts, nFichas, nSlot, SP, COIN, OT, alpha=GRASP_alpha, VERSION="C", hablar=False);
    inicial = GRASP(surgeon, second, patient, room, day, slot, AOR, I, dictCosts, nFichas, nSlot, SP, COIN, OT, alpha=GRASP_alpha, modo=1, VERSION="C", hablar=False);

    start_time = time.time()
    solutions = [];
    for ejec in range(5):
        best_solution, stats = metaheuristic(inicial, max_iter=max_iter, destruct_type=destruct_type, destruct=destruct, temp_inicial=temp_inicial, alpha=alpha,
                                            prob_CambiarPrimarios=prob_CambiarPrimarios, prob_CambiarSecundarios=prob_CambiarSecundarios,
                                            prob_MoverPaciente_bloque=prob_MoverPaciente_bloque, prob_MoverPaciente_dia=prob_MoverPaciente_dia,
                                            prob_EliminarPaciente=prob_EliminarPaciente, prob_AgregarPaciente_1=prob_AgregarPaciente_1, prob_AgregarPaciente_2=prob_AgregarPaciente_2,
                                            prob_DestruirAgregar10=prob_DestruirAgregar10,
                                            prob_MejorarAfinidad_primario=prob_MejorarAfinidad_primario, prob_MejorarAfinidad_secundario=prob_MejorarAfinidad_secundario,
                                            prob_AdelantarDia=prob_AdelantarDia, prob_MejorOR=prob_MejorOR,
                                            prob_AdelantarTodos=prob_AdelantarTodos,
                                            prob_CambiarPaciente1=prob_CambiarPaciente1, prob_CambiarPaciente2=prob_CambiarPaciente2, prob_CambiarPaciente3=prob_CambiarPaciente3,
                                            prob_Pert=prob_Pert, prob_Busq=prob_Busq, semilla=ejec, GRASP_alpha=GRASP_alpha, 
                                            elite_size=elite_size, prob_elite=prob_elite, prob_GRASP1=prob_GRASP1, prob_GRASP2=prob_GRASP2, prob_GRASP3=prob_GRASP3);
        solutions.append(EvalAllORs(best_solution[0], VERSION="C"));
    elapsed = time.time() - start_time
    #final_cost = EvalAllORs(best_solution[0], VERSION="C")
    #print(final_cost)
    print(np.mean(solutions))

if __name__ == "__main__":
    main()