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

import perturbations
importlib.reload(perturbations)
from perturbations import (
    CambiarPrimarios, CambiarSecundarios, MoverPaciente_bloque, MoverPaciente_dia,
    EliminarPaciente, AgregarPaciente_1, AgregarPaciente_2, DestruirAgregar10,
    DestruirAfinidad_Todos, DestruirAfinidad_Uno, PeorOR, AniquilarAfinidad
)

import algorithm._localsearches as _localsearches
importlib.reload(_localsearches)
from algorithm._localsearches import (
    MejorarAfinidad_primario, MejorarAfinidad_secundario, AdelantarDia, MejorOR,
    AdelantarTodos, CambiarPaciente1, CambiarPaciente2, CambiarPaciente3,
    CambiarPaciente4, CambiarPaciente5
)

import initial_solutions
importlib.reload(initial_solutions)
from initial_solutions import normal, GRASP

testing=False;
parametroFichas=0.11;
entrada="etapa1";
version="C";
solucion_inicial=True;

PERTURBATIONS=[
    CambiarPrimarios, CambiarSecundarios, MoverPaciente_bloque, MoverPaciente_dia,
    EliminarPaciente, AgregarPaciente_1, AgregarPaciente_2, DestruirAgregar10,
    DestruirAfinidad_Todos, DestruirAfinidad_Uno, PeorOR, AniquilarAfinidad
];

LOCAL_SEARCHES=[
    MejorarAfinidad_primario, MejorarAfinidad_secundario, AdelantarDia, MejorOR,
    AdelantarTodos, CambiarPaciente1, CambiarPaciente2, CambiarPaciente3,
    CambiarPaciente4, CambiarPaciente5
];

def EvalAllORs(sol, VERSION="C"):
    fichas=[[nFichas*(d+1) for d in range(len(day))] for s in surgeon];
    pacientes, primarios, secundarios=sol;
    def evalSchedule(pacientes, primarios, secundarios, or_id):
        bloques_por_paciente={}; penalizaciones=0; score_or=0;
        for p_idx in range(len(pacientes)):
            if pacientes[p_idx]!=-1:
                o_p, d_p, t_p=decompress(pacientes[p_idx]);
                if o_p==or_id:
                    duracion=OT[p_idx]; prioridad_paciente=I[(p_idx,d_p)];
                    s=primarios[pacientes[p_idx]]; a=secundarios[pacientes[p_idx]];
                    if p_idx not in bloques_por_paciente:
                        bloques_por_paciente[p_idx]=[];
                        score_or+=1000*prioridad_paciente;
                        s_idx=surgeon.index(s);
                        cost=dictCosts[(s,a,pacientes[p_idx])];
                        for d_aux in range(d_p,nDays):
                            fichas[s_idx][d_aux]-=cost;
                    for b in range(int(duracion)):
                        t_actual=t_p+b;
                        bloque_horario=compress(o_p,d_p,t_actual);
                        bloques_por_paciente[p_idx].append(bloque_horario);
                        if SP[p_idx][s]!=1: penalizaciones+=10;
                        if s==a: penalizaciones+=10;
        for paciente_id,bloques in bloques_por_paciente.items():
            bloques.sort(); duracion=OT[paciente_id];
            if len(bloques)!=duracion: penalizaciones+=50*len(bloques);
            if not all(bloques[i]+1==bloques[i+1] for i in range(len(bloques)-1)): penalizaciones+=100*len(bloques);
        score_or-=10*penalizaciones;
        return score_or;
    puntaje=0;
    for or_id in room:
        puntaje+=evalSchedule(pacientes,primarios,secundarios,or_id);
    for s_idx,_ in enumerate(surgeon):
        for d_idx in range(nDays):
            if fichas[s_idx][d_idx]<0: puntaje-=100*abs(fichas[s_idx][d_idx]);
    if VERSION=="C":
        def multiplicador(day_idx): return nDays//(day_idx+1);
        for s_idx,_ in enumerate(surgeon):
            for d_idx in range(nDays):
                puntaje-=fichas[s_idx][d_idx]*multiplicador(d_idx);
    return 1-(puntaje/bks);

def compress(o,d,t): return o*nSlot*nDays+d*nSlot+t;
def decompress(val):
    o=val//(nSlot*nDays); temp=val%(nSlot*nDays); d=temp//nSlot; t=temp%nSlot;
    return o,d,t;

def WhichExtra(o,t,d,e): return int(extras[o][t][d%5][e]);

def load_data_and_config():
    global dfSurgeon, dfSecond, dfRoom, dfType, dfPatient, extras;
    global patient, surgeon, second, room, day, nSlot, nRooms, slot;
    global nFichas, OT, AOR, COIN, Ex, I, SP, dictCosts;
    dfSurgeon=pd.read_excel("../input/MAIN_SURGEONS.xlsx",sheet_name='surgeon',converters={'n°':int},index_col=[0]);
    dfSecond=pd.read_excel("../input/SECOND_SURGEONS.xlsx",sheet_name='second',converters={'n°':int},index_col=[0]);
    dfRoom=pd.read_excel("../input/ROOMS.xlsx",sheet_name='room',converters={'n°':int},index_col=[0]);
    dfType=pd.read_excel("../input/PROCESS_TYPE.xls",sheet_name='Process Type',converters={'n°':int},index_col=[0]);
    if typePatients=="low": dfPatient=pd.read_csv("../input/LowPriority.csv");
    elif typePatients=="high": dfPatient=pd.read_csv("../input/HighPriority.csv");
    else: dfPatient=pd.read_csv("../input/AllPriority.csv");
    dfPatient=dfPatient.iloc[:nPatients];
    extras=[];
    def load_OR_dist(sheet_name):
        df_=pd.read_excel("../input/DIST_OR_EXT.xlsx",sheet_name=sheet_name,converters={'n°':int},index_col=[0]);
        df_=df_.astype(str).values.tolist(); ext_=[];
        for i in range(len(df_)): ext_.append(df_[i].copy());
        for i in range(len(df_[0])):
            part=df_[0][i].split(";"); df_[0][i]=part[0]; ext_[0][i]=part[1:];
            part2=df_[1][i].split(";"); df_[1][i]=part2[0]; ext_[1][i]=part2[1:];
        return df_,ext_;
    dfdisORA,extraA=load_OR_dist('A');
    dfdisORB,extraB=load_OR_dist('B');
    dfdisORC,extraC=load_OR_dist('C');
    dfdisORD,extraD=load_OR_dist('D');
    extras.extend([extraA,extraB,extraC,extraD]);
    dfdisAffi=pd.read_excel("../input/AFFINITY_EXT.xlsx",sheet_name='Hoja1',converters={'n°':int},index_col=[0]);
    dfdisAffiDiario=pd.read_excel("../input/AFFINITY_DIARIO.xlsx",sheet_name='Dias',converters={'n°':int},index_col=[0]);
    dfdisAffiBloque=pd.read_excel("../input/AFFINITY_DIARIO.xlsx",sheet_name='Bloques',converters={'n°':int},index_col=[0]);
    patient=[p for p in range(nPatients)];
    surgeon=[s for s in range(nSurgeons)];
    second=[a for a in range(nSurgeons)];
    room=[o for o in range(len(dfRoom))];
    day=[d for d in range(nDays)];
    nSlot=16; nRooms=len(room); slot=[t for t in range(nSlot)];
    I=np.ones((nPatients,nDays),dtype=int);
    for p in patient:
        for d in day: I[(p,d)]=1+dfPatient.iloc[p]["espera"]*dfPatient.iloc[p]["edad"]*0.0001/(d+1);
    COIN=np.zeros((nSurgeons,nSurgeons),dtype=int);
    for s in surgeon:
        for a in second:
            COIN[(s,a)]=1 if dfSurgeon.iloc[s][0]==dfSecond.iloc[a][0] else 0;
    dictCosts={};
    num_ext=2; extra=[]; dfExtra=[];
    for i in range(num_ext):
        aux=pd.read_excel("../input/AFFINITY_EXT.xlsx",sheet_name='Extra'+str(i+1),converters={'n°':int},index_col=[0]);
        extra.append(len(aux)); dfExtra.append(aux);
    Ex=[np.ones((nSurgeons,extra[i]),dtype=float) for i in range(num_ext)];
    for i in range(num_ext):
        for s in surgeon:
            for e in range(extra[i]): Ex[i][(s,e)]=dfExtra[i].iloc[e][s+1];
    for s in surgeon:
        for a in second:
            for _ in range(nSlot*nDays*len(room)):
                o,d,t=decompress(_);
                dictCosts[(s,a,_)]=int(dfdisAffi.iloc[a][s+1]+dfdisAffiDiario.iloc[d%5][s+1]+sum(Ex[i][(s,WhichExtra(o,t//(nSlot//2),d,i)-1)] for i in range(num_ext))+dfdisAffiBloque.iloc[t//(nSlot//2)][s+1]);
    SP=np.zeros((nPatients,nSurgeons),dtype=int);
    process=[t for t in range(len(dfType))];
    def busquedaType(especialidad):
        indice=0;
        for i in range(len(process)):
            if especialidad==dfType.iloc[i][0]: indice=i;
        return indice;
    for p in patient:
        for s in surgeon:
            if busquedaType(dfPatient.iloc[p]["especialidad"])==busquedaType(dfSurgeon.iloc[s][9]): SP[p][s]=1;
    OT=np.zeros(nPatients,dtype=int);
    for p in patient: OT[p]=int(dfPatient.iloc[p]["tipo"]);
    nFichas=int((parametroFichas*4*nSlot*len(room)*2*3)/(len(surgeon)**0.5));

def weighted_choice(items, weights):
    total=sum(weights); r=random.uniform(0,total); upto=0;
    for i,w in enumerate(weights):
        upto+=w;
        if upto>=r: return items[i];

def sa(initial_solution, iterations, temp_inicial, alpha, pert_probs, ls_probs, seed):
    random.seed(seed);
    current=copy.deepcopy(initial_solution); current_cost=EvalAllORs(current[0], VERSION="C");
    best=copy.deepcopy(current); best_cost=current_cost;
    temp=temp_inicial;
    for _ in range(iterations):
        neighbour=copy.deepcopy(current);
        pert_fn=weighted_choice(PERTURBATIONS, pert_probs);
        neighbour=pert_fn(neighbour, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays);
        ls_fn=weighted_choice(LOCAL_SEARCHES, ls_probs);
        neighbour=ls_fn(neighbour, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays);
        neigh_cost=EvalAllORs(neighbour[0], VERSION="C");
        delta=neigh_cost-current_cost;
        if delta<0 or random.random()<math.exp(-delta/temp):
            current=copy.deepcopy(neighbour); current_cost=neigh_cost;
            if neigh_cost<best_cost:
                best=copy.deepcopy(neighbour); best_cost=neigh_cost;
        temp*=alpha;
        if temp<1e-6: break;
    return best;

def main():
    parser=argparse.ArgumentParser();
    parser.add_argument("instance_file");
    parser.add_argument("--iterations",type=int,default=5000);
    parser.add_argument("--temp_inicial",type=float,default=800.0);
    parser.add_argument("--alpha",type=float,default=0.99);
    parser.add_argument("--seed",type=int,default=258);
    for p in ["CambiarPrimarios","CambiarSecundarios","MoverPaciente_bloque","MoverPaciente_dia",
              "EliminarPaciente","AgregarPaciente_1","AgregarPaciente_2","DestruirAgregar10",
              "DestruirAfinidad_Todos","DestruirAfinidad_Uno","PeorOR","AniquilarAfinidad"]:
        parser.add_argument(f"--prob_{p}",type=float,default=10.0);
    for s in ["MejorarAfinidad_primario","MejorarAfinidad_secundario","AdelantarDia","MejorOR",
              "AdelantarTodos","CambiarPaciente1","CambiarPaciente2","CambiarPaciente3",
              "CambiarPaciente4","CambiarPaciente5"]:
        parser.add_argument(f"--prob_{s}",type=float,default=10.0);
    args=parser.parse_args();
    with open(args.instance_file,'r') as f: data=json.load(f);
    global typePatients, nPatients, nDays, nSurgeons, bks, nFichas, min_affinity, day, slot;
    typePatients=data["patients"]; nPatients=int(data["n_patients"]); nDays=int(data["days"]);
    nSurgeons=int(data["surgeons"]); bks=int(data["bks"]); nFichas=int(data["fichas"]);
    min_affinity=int(data["min_affinity"]); time_limit=int(data["time_limit"]);
    load_data_and_config();
    initial=GRASP(surgeon, second, patient, room, day, slot, AOR, I, dictCosts, nFichas, nSlot,
                  SP, COIN, OT, alpha=0.1, modo=1, VERSION="C", hablar=False);
    pert_probs=[getattr(args,f"prob_{p}") for p in ["CambiarPrimarios","CambiarSecundarios","MoverPaciente_bloque","MoverPaciente_dia",
              "EliminarPaciente","AgregarPaciente_1","AgregarPaciente_2","DestruirAgregar10",
              "DestruirAfinidad_Todos","DestruirAfinidad_Uno","PeorOR","AniquilarAfinidad"]];
    ls_probs=[getattr(args,f"prob_{s}") for s in ["MejorarAfinidad_primario","MejorarAfinidad_secundario","AdelantarDia","MejorOR",
              "AdelantarTodos","CambiarPaciente1","CambiarPaciente2","CambiarPaciente3",
              "CambiarPaciente4","CambiarPaciente5"]];
    best=sa(initial, args.iterations, args.temp_inicial, args.alpha, pert_probs, ls_probs, args.seed);
    print(EvalAllORs(best[0], VERSION="C"))

if __name__=="__main__":
    main()