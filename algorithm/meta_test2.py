#!/usr/bin/env python3
# metaheuristic.py

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
import argparse
from pathlib import Path
import collections

class CSVMetaCheckpoint:
    def __init__(self, secs, csv_path, instance):
        self.secs       = sorted(secs)
        self.targetfile = Path(csv_path)
        self.instance   = instance
        self.next_idx   = 0
        self.first      = not self.targetfile.exists()
        self.targetfile.parent.mkdir(parents=True, exist_ok=True)
        self.best_gap = float("inf");
        self.avg_gap_global = None;
        self.iter_best_global = 0;

    def update_best(self, iteration, gap):
        if gap < self.best_gap:
            self.best_gap = gap;
            self.iter_best_global = iteration;

    def notify(self, elapsed, best_gap, avg_gap, iteration, patients):
        self.avg_gap_global = avg_gap
        if self.next_idx >= len(self.secs) or elapsed < self.secs[self.next_idx]:
            return
        row = {
            "instance": self.instance,
            "time": elapsed,
            "best_gap": best_gap,
            "avg_gap":  avg_gap,
            "iterations": iteration,
            "iter_best": self.iter_best_global,
            "patients": patients
        }
        pd.DataFrame([row]).to_csv(
            self.targetfile,
            mode='a',
            index=False,
            header=self.first
        )
        self.first = False;
        self.next_idx += 1;

class CSVMetaAggregator:
    def __init__(self, secs, csv_path, instance):
        self.secs       = secs
        self.targetfile = Path(csv_path)
        self.instance   = instance
        self.data = [dict(best_gaps=[], avg_gaps=[], iterations=[], iter_bests=[], patients=[]) for _ in secs]
        self.first = not self.targetfile.exists()

    def add(self, idx, best_gap, avg_gap, iterations, iter_best, patients):
        self.data[idx]["best_gaps"].append(best_gap)
        self.data[idx]["avg_gaps"].append(avg_gap)
        self.data[idx]["iterations"].append(iterations)
        self.data[idx]["iter_bests"].append(iter_best)
        self.data[idx]["patients"].append(patients)
    def finalize(self):
        rows = []
        for idx, sec in enumerate(self.secs):
            bucket = self.data[idx]
            if not bucket["best_gaps"]:
                continue
            avg_gap = sum(bucket["avg_gaps"]) / len(bucket["avg_gaps"])
            iterations = int(sum(bucket["iterations"]) / len(bucket["iterations"]))
            best_gap = min(bucket["best_gaps"])
            best_idx = bucket["best_gaps"].index(best_gap)
            iter_best = bucket["iter_bests"][best_idx]
            patients = int(sum(bucket["patients"]) / len(bucket["patients"]));
            rows.append({
                "instance": self.instance,
                "time": sec,
                "best_gap": best_gap,
                "avg_gap":  avg_gap,
                "iterations": iterations,
                "iter_best": iter_best,
                "patients": patients
            })
        if rows:
            pd.DataFrame(rows).to_csv(
                self.targetfile,
                mode='a',
                index=False,
                header=self.first
            )
            self.first = False

class RunCheckpoint:
    def __init__(self, secs, aggregator):
        self.secs = secs
        self.agg  = aggregator
        self.next_idx = 0
        self.iter_best_global = 0

    def update_best(self, iteration, gap):
        self.iter_best_global = iteration

    def notify(self, elapsed, best_gap, avg_gap, iteration, patients):
        if self.next_idx >= len(self.secs) or elapsed < self.secs[self.next_idx]:
            return
        self.agg.add(self.next_idx, best_gap, avg_gap, iteration, self.iter_best_global, patients)
        self.next_idx += 1

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
    DestruirAgregar10,
    DestruirAfinidad_Todos,
    DestruirAfinidad_Uno,
    PeorOR,
    AniquilarAfinidad
)

from evaluation import EvalAllORs

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
    CambiarPaciente3,
    CambiarPaciente4,
    CambiarPaciente5
)

import initial_solutions
importlib.reload(initial_solutions)
from initial_solutions import (
    normal,
    GRASP,
    complete_random
)

testing = False
parametroFichas = 0.11
version = "C"
solucion_inicial = True
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------------


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
    #    with open(f"input/{entrada}.json") as file:
    #        data = json.load(file)
    #else:
    #    with open("input/inst_test.json") as file:
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

def destruir_OR(solution, OT, dictCosts, nSlot, nDays, room, day):
    surgeon_schedule = copy.deepcopy(solution[1]);
    or_schedule = copy.deepcopy(solution[2]);
    fichas = copy.deepcopy(solution[3]);
    pacientes, primarios, secundarios = copy.deepcopy(solution[0][0]), copy.deepcopy(solution[0][1]), copy.deepcopy(solution[0][2]);

    chosen_or = random.choice(room);
    chosen_day = random.choice(day);
    possible_times = [0, nSlot//2];
    possible_times = [t for t in possible_times if t < nSlot];
    chosen_time = random.choice(possible_times);
    patients_to_remove = set();
    start_slot = chosen_time;
    end_slot = min(chosen_time + nSlot // 2, nSlot);
    for t_aux in range(start_slot, end_slot):
        if t_aux < len(or_schedule[chosen_or][chosen_day]):
            p = or_schedule[chosen_or][chosen_day][t_aux];
            if p != -1:
                patients_to_remove.add(p);

    for p in patients_to_remove:
        if p not in pacientes or pacientes[p] == -1:
            continue
        start_block = pacientes[p];
        o_real, d_real, t_real = decompress(start_block);
        dur = OT[p];
        main_s = primarios[start_block];
        second_s = secundarios[start_block];
        cost_key = (main_s, second_s, start_block);
        cost = dictCosts[cost_key];

        for b in range(dur):
            current_block = start_block + b;
            current_t = t_real + b;
            if 0 <= o_real < len(or_schedule) and 0 <= d_real < len(or_schedule[o_real]) and 0 <= current_t < len(or_schedule[o_real][d_real]):
                 or_schedule[o_real][d_real][current_t] = -1
            if 0 <= main_s < len(surgeon_schedule) and 0 <= d_real < len(surgeon_schedule[main_s]) and 0 <= current_t < len(surgeon_schedule[main_s][d_real]):
                 if surgeon_schedule[main_s][d_real][current_t] == p:
                     surgeon_schedule[main_s][d_real][current_t] = -1
            if 0 <= second_s < len(surgeon_schedule) and 0 <= d_real < len(surgeon_schedule[second_s]) and 0 <= current_t < len(surgeon_schedule[second_s][d_real]):
                 if surgeon_schedule[second_s][d_real][current_t] == p:
                     surgeon_schedule[second_s][d_real][current_t] = -1

            primarios.pop(current_block, None)
            secundarios.pop(current_block, None)
        for d_aux in range(d_real, nDays):
            fichas_key = (main_s, d_aux)
            if fichas_key in fichas:
                 fichas[fichas_key] += cost;
        pacientes[p] = -1
    return ((pacientes, primarios, secundarios), surgeon_schedule, or_schedule, fichas)

def metaheuristic(
        inicial, report_secs=[30], listener=None, destruct_type=1, destruct=200, temp_inicial=500.0, alpha=0.99,
        prob_CambiarPrimarios=15, prob_CambiarSecundarios=15, prob_MoverPaciente_bloque=20, prob_MoverPaciente_dia=10,
        prob_EliminarPaciente=20, prob_AgregarPaciente_1=19, prob_AgregarPaciente_2=19, prob_DestruirAgregar10=2,
        prob_DestruirAfinidad_Todos=2, prob_DestruirAfinidad_Uno=2, prob_PeorOR=2, prob_AniquilarAfinidad=5,
        prob_MejorarAfinidad_primario=20, prob_MejorarAfinidad_secundario=20, prob_AdelantarDia=29,
        prob_MejorOR=29, prob_AdelantarTodos=2, prob_CambiarPaciente1=10, prob_CambiarPaciente2=10, 
        prob_CambiarPaciente3=10, prob_CambiarPaciente4=10, prob_CambiarPaciente5=10,
        prob_DestruirOR=0.2, prob_elite=0.3, prob_GRASP=0.3, prob_normal=0.2,
        prob_Pert=1, prob_Busq=1, ils_extra=0.05, BusqTemp="yes", semilla=258, GRASP_alpha=0.1, elite_size=5,
        prob_GRASP1=0.3, prob_GRASP2=0.3, prob_GRASP3=0.4,
        acceptance_criterion="SA", tabu=False, tabulen=10, ini_random=0.05):
    random.seed(semilla);
    initial_time = time.time();
    report_secs_sorted = sorted(report_secs);
    last_report = report_secs_sorted[-1];
    next_report_idx = 0
    iter_best = 0

    initial_sol = inicial[0];
    surgeon_schedule = inicial[1];
    or_schedule = inicial[2];
    fichas = inicial[3];

    # [counts, improves, prob]
    metadata_pert = {"CambiarPrimarios": [0, 0, prob_CambiarPrimarios], "CambiarSecundarios": [0, 0, prob_CambiarSecundarios],
                    "MoverPaciente_bloque": [0, 0, prob_MoverPaciente_bloque], "MoverPaciente_dia": [0, 0, prob_MoverPaciente_dia],
                    "EliminarPaciente": [0, 0, prob_EliminarPaciente], "AgregarPaciente_1": [0, 0, prob_AgregarPaciente_1], "AgregarPaciente_2": [0, 0, prob_AgregarPaciente_2],
                    "DestruirAgregar10": [0, 0, prob_DestruirAgregar10], "DestruirAfinidad_Todos": [0, 0, prob_DestruirAfinidad_Todos], 
                    "DestruirAfinidad_Uno": [0, 0, prob_DestruirAfinidad_Uno], "PeorOR": [0, 0, prob_PeorOR], "AniquilarAfinidad": [0, 0, prob_AniquilarAfinidad], "NoOp": [0, 0, 0]};
    metadata_search = {"MejorarAfinidad_primario": [0, 0, prob_MejorarAfinidad_primario], "MejorarAfinidad_secundario": [0, 0, prob_MejorarAfinidad_secundario],
                       "AdelantarDia": [0, 0, prob_AdelantarDia], "MejorOR": [0, 0, prob_MejorOR], "AdelantarTodos": [0, 0, prob_AdelantarTodos], 
                       "CambiarPaciente1": [0, 0, prob_CambiarPaciente1], "CambiarPaciente2": [0, 0, prob_CambiarPaciente2], 
                       "CambiarPaciente3": [0, 0, prob_CambiarPaciente3], "CambiarPaciente4": [0, 0, prob_CambiarPaciente4], 
                       "CambiarPaciente5": [0, 0, prob_CambiarPaciente5], "NoOp": [0, 0, 0]};

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
        return sol, "NoOp";

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
        return sol, "NoOp";

    mejores_sols = [((initial_sol[0].copy(), initial_sol[1].copy(), initial_sol[2].copy()), surgeon_schedule.copy(), or_schedule.copy(), fichas.copy())];
    best_solution = ((initial_sol[0].copy(), initial_sol[1].copy(), initial_sol[2].copy()), surgeon_schedule.copy(), or_schedule.copy(), fichas.copy());
    best_sol = (best_solution[0][0].copy(), best_solution[0][1].copy(), best_solution[0][2].copy());
    best_cost = EvalAllORs(best_sol, VERSION=version,
                    hablar=False,
                    nFichas_val=nFichas,
                    day_py=day,
                    surgeon_py=surgeon,
                    room_py=room,
                    OT_obj=OT,
                    I_obj=I,
                    dictCosts_obj=dictCosts,
                    nDays_val=nDays,
                    nSlot_val=nSlot,
                    SP_obj=SP,
                    bks=bks
                );
    if listener:
        listener.update_best(0, best_cost);
    elite_pool = [(best_cost, copy.deepcopy(best_solution))];
    current_sol = ((best_sol[0].copy(), best_sol[1].copy(), best_sol[2].copy()), surgeon_schedule.copy(), or_schedule.copy(), fichas.copy());
    current_cost = best_cost;
    T = temp_inicial;
    r = 0;
    d_ = 0;
    tabulist = collections.deque(maxlen=tabulen) if tabu else None

    if BusqTemp == "yes":
        BusqTemp = 1;
    else:
        BusqTemp = 0;

    sum_gap = 0.0;
    count_iter = 0;
    i = 0
    while True:
        i += 1
        new_sol, last_p = Perturbar(current_sol);
        if BusqTemp == 0:
            if random.uniform(0, 1) < prob_Busq:
                new_sol, last_s = BusquedaLocal(new_sol);
            else:
                new_sol, last_s = copy.deepcopy(current_sol), "NoOp";
        
        else:
            if random.uniform(0, 1) < prob_Busq * (1 - T/temp_inicial):
                new_sol, last_s = BusquedaLocal(new_sol);
            else:
                new_sol, last_s = copy.deepcopy(current_sol), "NoOp";
        new_cost = EvalAllORs(new_sol[0], VERSION=version,
                    hablar=False,
                    nFichas_val=nFichas,
                    day_py=day,
                    surgeon_py=surgeon,
                    room_py=room,
                    OT_obj=OT,
                    I_obj=I,
                    dictCosts_obj=dictCosts,
                    nDays_val=nDays,
                    nSlot_val=nSlot,
                    SP_obj=SP,
                    bks=bks
                );
        cur_gap = best_cost;
        sum_gap += cur_gap;
        count_iter += 1;
        delta = current_cost - new_cost;

        istabu_reject = False
        sig = None
        if tabu and tabulist is not None:
            pac_assign = new_sol[0][0]
            sig = tuple(pac_assign) if isinstance(pac_assign, list) else pac_assign
            # Reject if solution is tabu and not improving the best cost
            if sig in tabulist and new_cost >= best_cost:
                istabu_reject = True
        if istabu_reject:
            d_ += 1
            T *= alpha
            continue

        ac = acceptance_criterion.lower();
        if ac == "no":
            if delta > 0:
                metadata_pert[last_p][1] += 1;
                metadata_search[last_s][1] += 1;
                current_sol = copy.deepcopy(new_sol);
                if tabu and sig is not None and tabulist is not None:
                    tabulist.append(sig)
                current_cost = new_cost;
                if new_cost < best_cost:
                    best_solution = copy.deepcopy(current_sol);
                    best_cost = current_cost;
                    elite_pool.append((best_cost, copy.deepcopy(best_solution)));
                    elite_pool.sort(key=lambda x: x[0], reverse=False);
                    elite_pool = elite_pool[:elite_size];
                    if 'iter_best' not in locals():
                        iter_best = i
                    if listener:
                        listener.update_best(i, best_cost);

        elif ac == "sa":
            if delta > 0 or random.random() < math.exp(delta / T):
                metadata_pert[last_p][1] += 1;
                metadata_search[last_s][1] += 1;
                current_sol = copy.deepcopy(new_sol);
                if tabu and sig is not None and tabulist is not None:
                    tabulist.append(sig)
                current_cost = new_cost;
                if new_cost < best_cost:
                    best_solution = copy.deepcopy(current_sol);
                    best_cost = current_cost;
                    elite_pool.append((best_cost, copy.deepcopy(best_solution)));
                    elite_pool.sort(key=lambda x: x[0], reverse=False);
                    elite_pool = elite_pool[:elite_size];
                    if 'iter_best' not in locals():
                        iter_best = i
                    if listener:
                        listener.update_best(i, best_cost);
        elif ac == "ils":
            if new_cost < (1 + ils_extra) *best_cost:
                metadata_pert[last_p][1] += 1;
                metadata_search[last_s][1] += 1;
                current_sol = copy.deepcopy(new_sol);
                if tabu and sig is not None and tabulist is not None:
                    tabulist.append(sig)
                current_cost = new_cost;
                best_solution = copy.deepcopy(current_sol);
                best_cost = current_cost;
                elite_pool.append((best_cost, copy.deepcopy(best_solution)));
                elite_pool.sort(key=lambda x: x[0], reverse=False);
                elite_pool = elite_pool[:elite_size];
                if 'iter_best' not in locals():
                    iter_best = i
                if listener:
                    listener.update_best(i, best_cost);
            else:
                current_sol = copy.deepcopy(best_solution);
                current_cost = best_cost;
        else:
            raise ValueError(f"Unknown acceptance criterion: {acceptance_criterion}");

        T *= alpha;
        if d_ >= destruct and destruct_type != 0: 
            mejores_sols.append(copy.deepcopy(current_sol));
            probab = random.choices([1, 2, 3, 4, 5], weights=[prob_DestruirOR, prob_elite, prob_GRASP, prob_normal, ini_random])[0];
            if probab == 1:
                current_sol = destruir_OR(current_sol, OT, dictCosts, nSlot, nDays, room, day);
                current_cost = EvalAllORs(current_sol[0], VERSION=version,
                    hablar=False,
                    nFichas_val=nFichas,
                    day_py=day,
                    surgeon_py=surgeon,
                    room_py=room,
                    OT_obj=OT,
                    I_obj=I,
                    dictCosts_obj=dictCosts,
                    nDays_val=nDays,
                    nSlot_val=nSlot,
                    SP_obj=SP,
                    bks=bks
                );
            elif probab == 2:
                _, chosen_elite_sol = random.choice(elite_pool);
                current_sol = copy.deepcopy(chosen_elite_sol);
                current_cost = EvalAllORs(current_sol[0], VERSION=version,
                    hablar=False,
                    nFichas_val=nFichas,
                    day_py=day,
                    surgeon_py=surgeon,
                    room_py=room,
                    OT_obj=OT,
                    I_obj=I,
                    dictCosts_obj=dictCosts,
                    nDays_val=nDays,
                    nSlot_val=nSlot,
                    SP_obj=SP,
                    bks=bks
                );
            elif probab == 3:
                pick = random.choices([1, 2, 3], weights=[prob_GRASP1, prob_GRASP2, prob_GRASP3])[0];
                current_sol = GRASP(surgeon, second, patient, room, day, slot, AOR, I, dictCosts, nFichas, nSlot, SP, COIN, OT, alpha=GRASP_alpha, modo=pick, VERSION="C", hablar=False);
                current_cost = EvalAllORs(current_sol[0], VERSION=version,
                    hablar=False,
                    nFichas_val=nFichas,
                    day_py=day,
                    surgeon_py=surgeon,
                    room_py=room,
                    OT_obj=OT,
                    I_obj=I,
                    dictCosts_obj=dictCosts,
                    nDays_val=nDays,
                    nSlot_val=nSlot,
                    SP_obj=SP,
                    bks=bks
                );
            elif probab == 4: 
                current_sol = normal(surgeon, second, patient, room, day, slot, AOR, I, dictCosts, nFichas, nSlot, SP, COIN, OT, VERSION="C", hablar=False);
                current_cost = EvalAllORs(current_sol[0], VERSION=version,
                    hablar=False,
                    nFichas_val=nFichas,
                    day_py=day,
                    surgeon_py=surgeon,
                    room_py=room,
                    OT_obj=OT,
                    I_obj=I,
                    dictCosts_obj=dictCosts,
                    nDays_val=nDays,
                    nSlot_val=nSlot,
                    SP_obj=SP,
                    bks=bks
                );
            else:
                current_sol = complete_random(surgeon, second, patient, room, day, slot, AOR, I, dictCosts, nFichas, nSlot, SP, COIN, OT, VERSION="C", hablar=False);
                current_cost = EvalAllORs(current_sol[0], VERSION=version,
                    hablar=False,
                    nFichas_val=nFichas,
                    day_py=day,
                    surgeon_py=surgeon,
                    room_py=room,
                    OT_obj=OT,
                    I_obj=I,
                    dictCosts_obj=dictCosts,
                    nDays_val=nDays,
                    nSlot_val=nSlot,
                    SP_obj=SP,
                    bks=bks
                );
            T = temp_inicial;
            d_ = 0;
        d_ += 1;
        current_time = time.time();
        elapsed = current_time - initial_time
        if next_report_idx < len(report_secs_sorted) and elapsed >= report_secs_sorted[next_report_idx]:
            best_gap = best_cost
            avg_gap  = sum_gap / count_iter if count_iter else best_gap;
            patients = sum(1 for p in best_solution[0][0] if p != -1)
            if listener:
                listener.notify(elapsed, best_gap, avg_gap, i, patients)
            else:
                print(f"[{elapsed/60:.1f} min] gap = {best_gap}")
            next_report_idx += 1
        if current_time - initial_time >= last_report:
            mejores_sols.append(copy.deepcopy(current_sol));
            break;
    
    total_iters = i;
    pacientes_best = best_solution[0][0]
    num_sched = sum(1 for p in pacientes_best if p != -1)
    mejores_sols.append(best_solution);

    mejor_costo, mejor = elite_pool[0];
    return mejor, (lista_evaluacion, lista_iteracion, metadata_pert, metadata_search, total_iters, iter_best, num_sched);

# ------------------------------------------------------------------------------------
# 3. MAIN
# ------------------------------------------------------------------------------------
def main():
    global typePatients, nPatients, nSurgeons, nDays, min_affinity, time_limit, bks
    parser = argparse.ArgumentParser()
    parser.add_argument("--destruct", type=int, default=200)
    parser.add_argument("--temp_inicial", type=float, default=800.0)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--prob_CambiarPrimarios", type=float, default=0.15)
    parser.add_argument("--prob_CambiarSecundarios", type=float, default=0.15)
    parser.add_argument("--prob_MoverPaciente_bloque", type=float, default=0.15)
    parser.add_argument("--prob_MoverPaciente_dia", type=float, default=0.15)
    parser.add_argument("--prob_EliminarPaciente", type=float, default=0.15)
    parser.add_argument("--prob_AgregarPaciente_1", type=float, default=0.15)
    parser.add_argument("--prob_AgregarPaciente_2", type=float, default=0.15)
    parser.add_argument("--prob_DestruirAgregar10", type=float, default=0.15)
    parser.add_argument("--prob_DestruirAfinidad_Todos", type=float, default=0.15)
    parser.add_argument("--prob_DestruirAfinidad_Uno", type=float, default=0.15)
    parser.add_argument("--prob_PeorOR", type=float, default=0.15)
    parser.add_argument("--prob_AniquilarAfinidad", type=float, default=0.05)
    parser.add_argument("--prob_MejorarAfinidad_primario", type=float, default=0.15)
    parser.add_argument("--prob_MejorarAfinidad_secundario", type=float, default=0.15)
    parser.add_argument("--prob_AdelantarDia", type=float, default=0.15)
    parser.add_argument("--prob_MejorOR", type=float, default=0.15)
    parser.add_argument("--prob_AdelantarTodos", type=float, default=0.15)
    parser.add_argument("--prob_CambiarPaciente1", type=float, default=0.15)
    parser.add_argument("--prob_CambiarPaciente2", type=float, default=0.15)
    parser.add_argument("--prob_CambiarPaciente3", type=float, default=0.15)
    parser.add_argument("--prob_CambiarPaciente4", type=float, default=0.15)
    parser.add_argument("--prob_CambiarPaciente5", type=float, default=0.15)
    parser.add_argument("--destruct_type", type=int, default=1)
    parser.add_argument("--prob_DestruirOR", type=float, default=0.2)
    parser.add_argument("--prob_elite", type=float, default=0.3)
    parser.add_argument("--prob_GRASP", type=float, default=0.3)
    parser.add_argument("--prob_normal", type=float, default=0.2)
    parser.add_argument("--prob_Busq", type=float, default=1.0)
    parser.add_argument("--BusqTemp", type=str, default="no")
    parser.add_argument("--ils_extra", type=float, default=0.05)
    parser.add_argument("--GRASP_alpha", type=float, default=0.1)
    parser.add_argument("--elite_size", type=int, default=5)
    parser.add_argument("--prob_GRASP1", type=float, default=0.3)
    parser.add_argument("--prob_GRASP2", type=float, default=0.3)
    parser.add_argument("--prob_GRASP3", type=float, default=0.4)
    parser.add_argument("--acceptance_criterion", type=str, default="SA")
    parser.add_argument("--tabu", type=int, default=0)
    parser.add_argument("--tabulen", type=int, default=10)
    parser.add_argument("--ini_random", type=float, default=0.05)
    parser.add_argument("--report_minutes", type=str, default="")

    args = parser.parse_args()
    instance_files = [f"../irace/instances/instance{i}.json" for i in range(1,4)];
    seeds = list(range(1))
    if args.report_minutes.strip():
        report_secs = [float(x)*60 for x in args.report_minutes.split(",") if x.strip()]
    else:
        report_secs = []

    overall_start = time.time()
    for instance_file in instance_files:
        instance_name = Path(instance_file).stem
        with open(instance_file, 'r') as f:
            data = json.load(f)

        typePatients = data["patients"]
        nPatients = int(data["n_patients"])
        nDays = int(data["days"])
        min_affinity = int(data["min_affinity"])
        nSurgeons = int(data["surgeons"])
        nFichas = int(data["fichas"])
        time_limit = int(data["time_limit"])
        bks = int(data["bks"])

        load_data_and_config()
        inicial = GRASP(surgeon, second, patient, room, day, slot, AOR, I, dictCosts,
                        nFichas, nSlot, SP, COIN, OT, alpha=args.GRASP_alpha, modo=1,
                        VERSION="C", hablar=False)

        aggregator = CSVMetaAggregator(report_secs,
                                       "metaheuristic_checkpoints.csv",
                                       instance_name)

        solutions = []
        all_iters = []
        all_best_iters = []
        all_num_sched = []

        for ejec in seeds:
            # fresh listener for this specific run
            listener = RunCheckpoint(report_secs, aggregator)
            best_solution, stats = metaheuristic(
                inicial,
                report_secs=report_secs,
                listener=listener,
                destruct_type=args.destruct_type,
                destruct=args.destruct,
                temp_inicial=args.temp_inicial,
                alpha=args.alpha,
                prob_CambiarPrimarios=args.prob_CambiarPrimarios,
                prob_CambiarSecundarios=args.prob_CambiarSecundarios,
                prob_MoverPaciente_bloque=args.prob_MoverPaciente_bloque,
                prob_MoverPaciente_dia=args.prob_MoverPaciente_dia,
                prob_EliminarPaciente=args.prob_EliminarPaciente,
                prob_AgregarPaciente_1=args.prob_AgregarPaciente_1,
                prob_AgregarPaciente_2=args.prob_AgregarPaciente_2,
                prob_DestruirAgregar10=args.prob_DestruirAgregar10,
                prob_DestruirAfinidad_Todos=args.prob_DestruirAfinidad_Todos,
                prob_DestruirAfinidad_Uno=args.prob_DestruirAfinidad_Uno,
                prob_PeorOR=args.prob_PeorOR,
                prob_AniquilarAfinidad=args.prob_AniquilarAfinidad,
                prob_MejorarAfinidad_primario=args.prob_MejorarAfinidad_primario,
                prob_MejorarAfinidad_secundario=args.prob_MejorarAfinidad_secundario,
                prob_AdelantarDia=args.prob_AdelantarDia,
                prob_MejorOR=args.prob_MejorOR,
                prob_AdelantarTodos=args.prob_AdelantarTodos,
                prob_CambiarPaciente1=args.prob_CambiarPaciente1,
                prob_CambiarPaciente2=args.prob_CambiarPaciente2,
                prob_CambiarPaciente3=args.prob_CambiarPaciente3,
                prob_CambiarPaciente4=args.prob_CambiarPaciente4,
                prob_CambiarPaciente5=args.prob_CambiarPaciente5,
                prob_DestruirOR=args.prob_DestruirOR,
                prob_elite=args.prob_elite,
                prob_GRASP=args.prob_GRASP,
                prob_normal=args.prob_normal,
                prob_Pert=1,
                prob_Busq=args.prob_Busq,
                BusqTemp=args.BusqTemp,
                ils_extra=args.ils_extra,
                semilla=ejec,
                GRASP_alpha=args.GRASP_alpha,
                elite_size=args.elite_size,
                prob_GRASP1=args.prob_GRASP1,
                prob_GRASP2=args.prob_GRASP2,
                prob_GRASP3=args.prob_GRASP3,
                acceptance_criterion=args.acceptance_criterion,
                tabu=args.tabu,
                tabulen=args.tabulen,
                ini_random=args.ini_random
            )

            _,_,_,_, avg_iter, best_iter, num_sched = stats
            solutions.append(EvalAllORs(best_solution[0], VERSION=version,
                    hablar=False,
                    nFichas_val=nFichas,
                    day_py=day,
                    surgeon_py=surgeon,
                    room_py=room,
                    OT_obj=OT,
                    I_obj=I,
                    dictCosts_obj=dictCosts,
                    nDays_val=nDays,
                    nSlot_val=nSlot,
                    SP_obj=SP,
                    bks=bks
                ));
            all_iters.append(avg_iter)
            all_best_iters.append(best_iter)
            all_num_sched.append(num_sched)

        aggregator.finalize()

        print(f"{instance_file}  |  mean_gap: {np.mean(solutions):.5f}")
    print(f"Total elapsed: {time.time()-overall_start:.1f} s")

if __name__ == "__main__":
    main()

'''
/opt/homebrew/Cellar/python@3.10/3.10.17/Frameworks/Python.framework/Versions/3.10/bin/python3.10 meta_test2.py --destruct 400 --temp_ini 1628.627 --alpha 0.998 --prob_CambiarPrimarios 0.5072 --prob_CambiarSecundarios 0.5246 --prob_MoverPaciente_bloque 0.1069 --prob_MoverPaciente_dia 0.1599 --prob_EliminarPaciente 0.2362 --prob_AgregarPaciente_1 0.6621 --prob_AgregarPaciente_2 0.9674 --prob_DestruirAgregar10 0.4148 --prob_DestruirAfinidad_Todos 0.1435 --prob_DestruirAfinidad_Uno 0.6722 --prob_PeorOR 0.2447 --prob_AniquilarAfinidad 0.6051 --prob_MejorarAfinidad_primario 0.6632 --prob_MejorarAfinidad_secundario 0.7004 --prob_AdelantarDia 0.5502 --prob_MejorOR 0.0744 --prob_AdelantarTodos 0.4141 --prob_CambiarPaciente1 0.8321 --prob_CambiarPaciente2 0.253 --prob_CambiarPaciente3 0.1407 --prob_CambiarPaciente4 0.5862 --prob_CambiarPaciente5 0.3964 --destruct_type 1 --prob_DestruirOR 0.4114 --prob_elite 0.8667 --prob_GRASP 0.1738 --prob_normal 0.1347 --prob_Busq 0.9298 --BusqTemp yes --GRASP_alpha 0.4942 --elite_size 7 --prob_GRASP1 0.3496 --prob_GRASP2 0.399 --prob_GRASP3 0.4853 --acceptance_criterion SA --tabu 0 --ini_random 0.4307 --report_minutes "0.3,1,1.5"
'''