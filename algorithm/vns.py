#!/usr/bin/env python3
import sys
import warnings
warnings.filterwarnings("ignore")
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

import perturbations
importlib.reload(perturbations)
from perturbations import (
    CambiarPrimarios,CambiarSecundarios,MoverPaciente_bloque,MoverPaciente_dia,
    EliminarPaciente,AgregarPaciente_1,AgregarPaciente_2,DestruirAgregar10,
    DestruirAfinidad_Todos,DestruirAfinidad_Uno,PeorOR,AniquilarAfinidad
)

import localsearches
importlib.reload(localsearches)
from localsearches import (
    MejorarAfinidad_primario,MejorarAfinidad_secundario,AdelantarDia,MejorOR,
    AdelantarTodos,CambiarPaciente1,CambiarPaciente2,CambiarPaciente3,
    CambiarPaciente4,CambiarPaciente5
)

import initial_solutions as initial_solutions
importlib.reload(initial_solutions)
from initial_solutions import normal,GRASP, complete_random

from evaluation import EvalAllORs

class CSVCheckpoint:
    def __init__(self,secs,csv_path,instance,aggregator=None):
        self.secs=sorted(secs);
        self.targetfile=Path(csv_path);
        self.instance=instance;
        self.next_idx=0;
        self.first=not self.targetfile.exists();
        self.targetfile.parent.mkdir(parents=True,exist_ok=True);
        self.gaps = []
        self.best_gap = float('inf')
        self.total_iterations = 0
        self.iter_best_global = 0
        self.aggregator = aggregator

    def notify(self, elapsed, gap, iteration, patients):
        # record history
        self.gaps.append(gap)

        if gap < self.best_gap:
            self.best_gap = gap
            self.iter_best_global = iteration

        if self.next_idx >= len(self.secs) or elapsed < self.secs[self.next_idx]:
            return

        avg_gap = sum(self.gaps) / len(self.gaps)

        row_best = self.best_gap
        row_avg  = avg_gap
        row_iter = iteration
        row_iter_best = self.iter_best_global

        if self.aggregator is not None:
            self.aggregator.add(self.next_idx,
                                row_best, row_avg,
                                row_iter, row_iter_best, patients)
        else:
            pd.DataFrame([{
                "instance": self.instance,
                "time": elapsed,
                "best_gap": row_best,
                "avg_gap":  row_avg,
                "iterations": row_iter,
                "iter_best": row_iter_best,
                "patients": patients
            }]).to_csv(
                self.targetfile,
                mode="a",
                index=False,
                header=self.first
            )
            self.first = False
        self.next_idx += 1


# Aggregator class for CSVCheckpoint
class CSVCheckpointAggregator:
    """Aggregate checkpoint stats from many runs then write averaged rows."""
    def __init__(self, secs, csv_path, instance):
        self.secs       = secs
        self.targetfile = Path(csv_path)
        self.instance   = instance
        self.data       = [dict(best_gaps=[], avg_gaps=[], iterations=[], iter_bests=[], patients=[]) for _ in secs]
        self.first      = not self.targetfile.exists()

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
            avg_gap    = sum(bucket["avg_gaps"]) / len(bucket["avg_gaps"])
            iterations = int(sum(bucket["iterations"]) / len(bucket["iterations"]))
            best_gap   = min(bucket["best_gaps"])
            best_idx   = bucket["best_gaps"].index(best_gap)
            iter_best  = bucket["iter_bests"][best_idx]
            patients   = int(sum(bucket["patients"])/len(bucket["patients"]))
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
                mode="a",
                index=False,
                header=self.first
            )
            self.first = False

testing=False;
parametroFichas=0.11;
version="C";
warnings.filterwarnings("ignore");

PERTURBATIONS=[
    CambiarPrimarios,CambiarSecundarios,MoverPaciente_bloque,MoverPaciente_dia,
    EliminarPaciente,AgregarPaciente_1,AgregarPaciente_2,DestruirAgregar10,
    DestruirAfinidad_Todos,DestruirAfinidad_Uno,PeorOR,AniquilarAfinidad
];

LOCAL_SEARCHES=[
    MejorarAfinidad_primario,MejorarAfinidad_secundario,AdelantarDia,MejorOR,
    AdelantarTodos,CambiarPaciente1,CambiarPaciente2,CambiarPaciente3,
    CambiarPaciente4,CambiarPaciente5
];

def compress(o,d,t): 
    return o*nSlot*nDays+d*nSlot+t
def decompress(val):
    o=val//(nSlot*nDays); 
    temp=val%(nSlot*nDays); 
    d=temp//nSlot; 
    t=temp%nSlot
    return o,d,t
def WhichExtra(o,t,d,e): 
    return int(extras[o][t][d%5][e])

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

def weighted_choice(items,weights):
    total=sum(weights);r=random.uniform(0,total);upto=0;
    for i,w in enumerate(weights):
        upto+=w;
        if upto>=r:return items[i];

def vns(initial_solution,k_max,pert_probs,ls_probs,seed,report_secs, listener=None, reset_iter=1000, iterations=None):
    random.seed(seed);
    best=copy.deepcopy(initial_solution);best_cost=EvalAllORs(best[0],VERSION=version,
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
                        bks=bks)
    k=1;
    it=0;
    rep=sorted(report_secs);
    nxt=0;
    start=time.time();
    stagnation = 0; 
    while True:
        it+=1;
        base=copy.deepcopy(best);
        pert_fn=PERTURBATIONS[(k-1)%len(PERTURBATIONS)];
        cand=pert_fn(base,surgeon,second,OT,I,SP,AOR,dictCosts,nSlot,nDays);
        ls_fn=weighted_choice(LOCAL_SEARCHES,ls_probs);
        cand=ls_fn(cand,surgeon,second,OT,I,SP,AOR,dictCosts,nSlot,nDays);
        c_cost=EvalAllORs(cand[0],VERSION=version,
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
                        bks=bks);
        if c_cost<best_cost:
            best=copy.deepcopy(cand);
            best_cost=c_cost;
            k=1;
            stagnation = 0;
        else:
            k += 1;
            if k > k_max: k = 1;
            stagnation += 1;
        if stagnation >= reset_iter:
            k = 1;
            stagnation = 0;
        elapsed=time.time()-start;
        if nxt<len(rep) and elapsed>=rep[nxt]:
            gap=best_cost;
            patients_scheduled = sum(1 for p in best[0][0] if p != -1);
            if listener:
                listener.notify(elapsed, gap, it, patients_scheduled);
            else:
                print(f"[{elapsed/60:.1f} min] gap = {gap}");
            nxt+=1;
        if nxt>=len(rep):break;
        if iterations is not None and len(rep)==0 and it>=iterations:break;
    return best;

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_max", type=int, default=5)
    parser.add_argument("--reset_iter", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=258)
    for p in ["CambiarPrimarios","CambiarSecundarios","MoverPaciente_bloque","MoverPaciente_dia",
              "EliminarPaciente","AgregarPaciente_1","AgregarPaciente_2","DestruirAgregar10",
              "DestruirAfinidad_Todos","DestruirAfinidad_Uno","PeorOR","AniquilarAfinidad"]:
        parser.add_argument(f"--prob_{p}", type=float, default=10.0)
    for s in ["MejorarAfinidad_primario","MejorarAfinidad_secundario","AdelantarDia","MejorOR",
              "AdelantarTodos","CambiarPaciente1","CambiarPaciente2","CambiarPaciente3",
              "CambiarPaciente4","CambiarPaciente5"]:
        parser.add_argument(f"--prob_{s}", type=float, default=10.0)
    parser.add_argument("--report_minutes", type=str, default="");
    args, _ = parser.parse_known_args();
    report_secs = [float(x)*60 for x in args.report_minutes.split(",") if x.strip()] if args.report_minutes.strip() else [];

    seeds = list(range(10))
    for idx in range(1, 16):
        instance_file = f"../irace/instances/instance{idx}.json"

        # load instance data
        with open(instance_file, "r") as f:
            data = json.load(f)
        global typePatients, nPatients, nDays, nSurgeons, bks, nFichas, min_affinity, time_limit
        global day, slot, surgeon, second, room, AOR, dictCosts, OT, I, SP, nSlot, COIN
        typePatients = data["patients"]
        nPatients = int(data["n_patients"])
        nDays = int(data["days"])
        nSurgeons = int(data["surgeons"])
        bks = int(data["bks"])
        nFichas = int(data["fichas"])
        min_affinity = int(data["min_affinity"])
        time_limit = int(data["time_limit"])

        load_data_and_config()

        aggregator = CSVCheckpointAggregator(report_secs,
                                             "vns_checkpoints.csv",
                                             f"instance{idx}")

        initial = GRASP(surgeon, second, patient, room, day, slot, AOR, I,
                        dictCosts, nFichas, nSlot, SP, COIN, OT,
                        alpha=0.1, modo=1, VERSION="C", hablar=False)

        pert_probs = [getattr(args, f"prob_{p}") for p in ["CambiarPrimarios","CambiarSecundarios",
                      "MoverPaciente_bloque","MoverPaciente_dia","EliminarPaciente","AgregarPaciente_1",
                      "AgregarPaciente_2","DestruirAgregar10","DestruirAfinidad_Todos",
                      "DestruirAfinidad_Uno","PeorOR","AniquilarAfinidad"]]
        ls_probs = [getattr(args, f"prob_{s}") for s in ["MejorarAfinidad_primario",
                   "MejorarAfinidad_secundario","AdelantarDia","MejorOR","AdelantarTodos",
                   "CambiarPaciente1","CambiarPaciente2","CambiarPaciente3",
                   "CambiarPaciente4","CambiarPaciente5"]]

        solutions = []
        for ejec in seeds:
            listener = CSVCheckpoint(report_secs,
                                     "vns_checkpoints.csv",
                                     f"instance{idx}",
                                     aggregator=aggregator)
            best = vns(initial,
                       args.k_max,
                       pert_probs,
                       ls_probs,
                       ejec,
                       report_secs,
                       listener=listener,
                       reset_iter=args.reset_iter,
                       iterations=args.iterations)
            solutions.append(EvalAllORs(best[0], VERSION=version,
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
                        bks=bks))
        aggregator.finalize()
        print(f"Instance {idx}: mean_cost = {-np.mean(solutions):.5f}, "
              f"mean_gap = {1 - (-np.mean(solutions)/bks):.5f}")
if __name__=="__main__":
    main();
#python vns.py --k_max 5 --reset_iter 1000 --iterations 50000 --seed 258 --report_minutes "0.1,0.3,0.5" --prob_CambiarPaciente5 10.0