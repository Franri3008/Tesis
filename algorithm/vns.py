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

import _localsearches
importlib.reload(_localsearches)
from _localsearches import (
    MejorarAfinidad_primario,MejorarAfinidad_secundario,AdelantarDia,MejorOR,
    AdelantarTodos,CambiarPaciente1,CambiarPaciente2,CambiarPaciente3,
    CambiarPaciente4,CambiarPaciente5
)

import initial_solutions
importlib.reload(initial_solutions)
from initial_solutions import normal,GRASP

class CSVCheckpoint:
    def __init__(self,secs,csv_path,instance):
        self.secs=sorted(secs);
        self.targetfile=Path(csv_path);
        self.instance=instance;
        self.next_idx=0;
        self.first=not self.targetfile.exists();
        self.targetfile.parent.mkdir(parents=True,exist_ok=True);
    def notify(self,elapsed,gap):
        if self.next_idx>=len(self.secs) or elapsed<self.secs[self.next_idx]:
            return;
        pd.DataFrame([{"instance":self.instance,"time":elapsed,"gap":gap}]).to_csv(
            self.targetfile,mode="a",index=False,header=self.first
        );
        self.first=False;
        self.next_idx+=1;

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

def EvalAllORs(sol,VERSION="C"):
    fichas=[[nFichas*(d+1) for d in range(len(day))] for s in surgeon];
    pacientes,primarios,secundarios=sol;
    def evalSchedule(pacientes,primarios,secundarios,or_id):
        bloques_por_paciente={};penalizaciones=0;score_or=0;
        for p_idx in range(len(pacientes)):
            if pacientes[p_idx]!=-1:
                o_p,d_p,t_p=decompress(pacientes[p_idx]);
                if o_p==or_id:
                    duracion=OT[p_idx];prioridad_paciente=I[(p_idx,d_p)];
                    s=primarios[pacientes[p_idx]];a=secundarios[pacientes[p_idx]];
                    if p_idx not in bloques_por_paciente:
                        bloques_por_paciente[p_idx]=[];
                        score_or+=1000*prioridad_paciente;
                        s_idx=surgeon.index(s);cost=dictCosts[(s,a,pacientes[p_idx])];
                        for d_aux in range(d_p,nDays):
                            fichas[s_idx][d_aux]-=cost;
                    for b in range(int(duracion)):
                        t_actual=t_p+b;bloque_horario=compress(o_p,d_p,t_actual);
                        bloques_por_paciente[p_idx].append(bloque_horario);
                        if SP[p_idx][s]!=1:penalizaciones+=10;
                        if s==a:penalizaciones+=10;
        for paciente_id,bloques in bloques_por_paciente.items():
            bloques.sort();duracion=OT[paciente_id];
            if len(bloques)!=duracion:penalizaciones+=50*len(bloques);
            if not all(bloques[i]+1==bloques[i+1] for i in range(len(bloques)-1)):
                penalizaciones+=100*len(bloques);
        score_or-=10*penalizaciones;
        return score_or;
    puntaje=0;
    for or_id in room:puntaje+=evalSchedule(pacientes,primarios,secundarios,or_id);
    for s_idx,_ in enumerate(surgeon):
        for d_idx in range(nDays):
            if fichas[s_idx][d_idx]<0:puntaje-=100*abs(fichas[s_idx][d_idx]);
    if VERSION=="C":
        def multiplicador(day_idx):return nDays//(day_idx+1);
        for s_idx,_ in enumerate(surgeon):
            for d_idx in range(nDays):
                puntaje-=fichas[s_idx][d_idx]*multiplicador(d_idx);
    return 1-(puntaje/bks);

def compress(o,d,t): 
    return o*nSlot*nDays+d*nSlot+t
def decompress(val):
    o=val//(nSlot*nDays); 
    temp=val%(nSlot*nDays); 
    d=temp//nSlot; 
    t=temp%nSlot
    return o,d,t
def WhichExtra(o,t,d,e): return int(extras[o][t][d%5][e])

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
    best=copy.deepcopy(initial_solution);best_cost=EvalAllORs(best[0],VERSION="C");
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
        c_cost=EvalAllORs(cand[0],VERSION="C");
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
            if listener:
                listener.notify(elapsed,gap);
            else:
                print(f"[{elapsed/60:.1f} min] gap = {gap}");
            nxt+=1;
        if nxt>=len(rep):break;
        if iterations is not None and len(rep)==0 and it>=iterations:break;
    return best;

def main():
    parser=argparse.ArgumentParser();
    parser.add_argument("instance_file");
    parser.add_argument("--k_max",type=int,default=5);
    parser.add_argument("--reset_iter", type=int, default=1000);
    parser.add_argument("--iterations",type=int,default=5000);
    parser.add_argument("--seed",type=int,default=258);
    for p in ["CambiarPrimarios","CambiarSecundarios","MoverPaciente_bloque","MoverPaciente_dia",
              "EliminarPaciente","AgregarPaciente_1","AgregarPaciente_2","DestruirAgregar10",
              "DestruirAfinidad_Todos","DestruirAfinidad_Uno","PeorOR","AniquilarAfinidad"]:
        parser.add_argument(f"--prob_{p}",type=float,default=10.0);
    for s in ["MejorarAfinidad_primario","MejorarAfinidad_secundario","AdelantarDia","MejorOR",
              "AdelantarTodos","CambiarPaciente1","CambiarPaciente2","CambiarPaciente3",
              "CambiarPaciente4","CambiarPaciente5"]:
        parser.add_argument(f"--prob_{s}",type=float,default=10.0);
    parser.add_argument("--report_minutes",type=str,default="");
    args=parser.parse_args();
    if args.report_minutes.strip():
        report_secs=[float(x)*60 for x in args.report_minutes.split(",") if x.strip()];
    else:
        report_secs=[];
    with open(args.instance_file,"r") as f:data=json.load(f);
    global typePatients,nPatients,nDays,nSurgeons,bks,nFichas,min_affinity,day,slot,surgeon,second,room,AOR,dictCosts,OT,I,SP,nSlot,COIN
    typePatients=data["patients"];nPatients=int(data["n_patients"]);nDays=int(data["days"]);
    nSurgeons=int(data["surgeons"]);bks=int(data["bks"]);nFichas=int(data["fichas"]);
    min_affinity=int(data["min_affinity"]);time_limit=int(data["time_limit"]);
    load_data_and_config();
    initial=GRASP(surgeon,second,patient,room,day,slot,AOR,I,dictCosts,nFichas,nSlot,SP,COIN,OT,alpha=0.1,modo=1,VERSION="C",hablar=False);
    pert_probs=[getattr(args,f"prob_{p}") for p in ["CambiarPrimarios","CambiarSecundarios","MoverPaciente_bloque","MoverPaciente_dia",
              "EliminarPaciente","AgregarPaciente_1","AgregarPaciente_2","DestruirAgregar10",
              "DestruirAfinidad_Todos","DestruirAfinidad_Uno","PeorOR","AniquilarAfinidad"]];
    ls_probs=[getattr(args,f"prob_{s}") for s in ["MejorarAfinidad_primario","MejorarAfinidad_secundario","AdelantarDia","MejorOR",
              "AdelantarTodos","CambiarPaciente1","CambiarPaciente2","CambiarPaciente3",
              "CambiarPaciente4","CambiarPaciente5"]];
    listener=CSVCheckpoint(report_secs,"vns_checkpoints.csv",args.instance_file);

    best = vns(initial,
           args.k_max,
           pert_probs,
           ls_probs,
           args.seed,
           report_secs,
           listener=listener,
           reset_iter=args.reset_iter,
           iterations=None);
    print(EvalAllORs(best[0],VERSION="C"));
    
if __name__=="__main__":
    main();
