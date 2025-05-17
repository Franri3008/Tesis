
testing = False;
parametroFichas = 0.11;
entrada = "instance1";
version = "C";
solucion_inicial = True;
print(f"Configurado con testing = {testing}. Versión: {version}. Entrada: {entrada}. Solución inicial: {solucion_inicial}")
#exe_loc = "C:/Program Files/IBM/ILOG/CPLEX_Studio221/cpoptimizer/bin/x64_win64/cpoptimizer.exe";
exe_loc = "/Applications/CPLEX_Studio2211/cpoptimizer/bin/arm64_osx/cpoptimizer";

from docplex.cp.config import context
context.solver.agent = 'local';
context.solver.local.execfile = "/Applications/CPLEX_Studio2211/cpoptimizer/bin/arm64_osx/cpoptimizer";

from docplex.cp.model import CpoModel

import warnings
import random
import json
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from timeit import default_timer
from pathlib import Path

import os
import sys
notebook_dir = os.getcwd();
os.chdir(notebook_dir)

warnings.filterwarnings("ignore")

if testing == False:
    with open(f"../irace/instances/{entrada}.json") as file:
        data = json.load(file);
else:
    with open("../input/inst_test.json") as file:
        data = json.load(file);
    
#config = data["configurations"];
global_checkpoint_csv_path = Path("../output/CPM/CPM_checkpoints.csv");
global_checkpoint_csv_path.parent.mkdir(parents=True, exist_ok=True);
if global_checkpoint_csv_path.exists():
    global_checkpoint_csv_path.unlink()

dfSolucion = pd.DataFrame();
dfSolucion.to_excel(f"../output/CPM/{entrada}_output.xlsx", index=False);
                    
dfSurgeon = pd.read_excel("../input/MAIN_SURGEONS.xlsx", sheet_name='surgeon', converters={'n°':int}, index_col=[0]);
dfSecond = pd.read_excel("../input/SECOND_SURGEONS.xlsx", sheet_name='second', converters={'n°':int}, index_col=[0]);
dfRoom = pd.read_excel("../input/ROOMS.xlsx", sheet_name='room', converters={'n°':int}, index_col=[0]);
dfType = pd.read_excel("../input/PROCESS_TYPE.xls", sheet_name='Process Type', converters={'n°':int}, index_col=[0]);
                    
extras = [];
dfdisORA = pd.read_excel("../input/DIST_OR_EXT.xlsx", sheet_name='A', converters={'n°':int}, index_col=[0]);
dfdisORA = dfdisORA.astype(str).values.tolist();
extraA = [];
for i in range(len(dfdisORA)):
    extraA.append(dfdisORA[i].copy());
for i in range(len(dfdisORA[0])):
    aux = dfdisORA[0][i].split(";");
    dfdisORA[0][i] = aux[0];
    extraA[0][i] = aux[1:];
    aux = dfdisORA[1][i];
    aux = aux.split(";");
    dfdisORA[1][i] = aux[0];
    extraA[1][i] = aux[1:];
    num_ext = len(extraA[1][i]);
extras.append(extraA);

dfdisORB = pd.read_excel("../input/DIST_OR_EXT.xlsx", sheet_name='B', converters={'n°':int}, index_col=[0]);
dfdisORB = dfdisORB.astype(str).values.tolist();
extraB = [];
for i in range(len(dfdisORB)):
    aux = dfdisORB[i].copy();
    extraB.append(aux);
for i in range(len(dfdisORB[0])):
    aux = dfdisORB[0][i];
    aux = aux.split(";");
    dfdisORB[0][i] = aux[0];
    extraB[0][i] = aux[1:];
    aux = dfdisORB[1][i];
    aux = aux.split(";");
    dfdisORB[1][i] = aux[0];
    extraB[1][i] = aux[1:];
extras.append(extraB);

dfdisORC = pd.read_excel("../input/DIST_OR_EXT.xlsx", sheet_name='C', converters={'n°':int}, index_col=[0]);
dfdisORC = dfdisORC.astype(str).values.tolist();
extraC = [];
for i in range(len(dfdisORC)):
    aux = dfdisORC[i].copy();
    extraC.append(aux);
for i in range(len(dfdisORC[0])):
    aux = dfdisORC[0][i];
    aux = aux.split(";");
    dfdisORC[0][i] = aux[0];
    extraC[0][i] = aux[1:];
    aux = dfdisORC[1][i];
    aux = aux.split(";");
    dfdisORC[1][i] = aux[0];
    extraC[1][i] = aux[1:];
extras.append(extraC);

dfdisORD = pd.read_excel("../input/DIST_OR_EXT.xlsx", sheet_name='D', converters={'n°':int}, index_col=[0]);
dfdisORD = dfdisORD.astype(str).values.tolist();
extraD = [];
for i in range(len(dfdisORD)):
    aux = dfdisORD[i].copy();
    extraD.append(aux);
for i in range(len(dfdisORD[0])):
    aux = dfdisORD[0][i];
    aux = aux.split(";");
    dfdisORD[0][i] = aux[0];
    extraD[0][i] = aux[1:];
    aux = dfdisORD[1][i];
    aux = aux.split(";");
    dfdisORD[1][i] = aux[0];
    extraD[1][i] = aux[1:];
extras.append(extraD);

dfdisAffi = pd.read_excel("../input/AFFINITY_EXT.xlsx", sheet_name='Hoja1', converters={'n°':int}, index_col=[0]);
dfdisAffiDiario = pd.read_excel("../input/AFFINITY_DIARIO.xlsx", sheet_name='Dias', converters={'n°':int}, index_col=[0]);
dfdisAffiBloque = pd.read_excel("../input/AFFINITY_DIARIO.xlsx", sheet_name='Bloques', converters={'n°':int}, index_col=[0]);
dfdisRank = pd.read_excel("../input/RANKING.xlsx", sheet_name = 'Hoja1', converters={'n°':int}, index_col=[0]);

extra = [];
dfExtra = [];
for i in range(num_ext):
    aux = pd.read_excel("../input/AFFINITY_EXT.xlsx", sheet_name='Extra'+str(i+1), converters = {'n°':int}, index_col=[0]);
    extra.append(len(aux));
    dfExtra.append(aux);

extrasCPM = [item for sublist in extras for item in sublist];

for i in range(2):
    extrasCPM = [item for sublist in extrasCPM for item in sublist];
    
extrasCPM = [int(item) for sublist in extrasCPM for item in sublist]

# Rankings para extras
dfRankExtra = [];
for i in range(num_ext):
    aux = pd.read_excel("../input/RANKING.xlsx", sheet_name='Extra'+str(i+1), converters = {'n°':int}, index_col=[0]);
    extra.append(len(aux));
    dfRankExtra.append(aux);

#-------------------------------------------------------------------------------------------------------------------------------#     
########################################################### MAIN LOOP ###########################################################
#-------------------------------------------------------------------------------------------------------------------------------#  

for iii in range(1, 16):
    entrada = f"instance{iii}";
    if testing == False:
        with open(f"../irace/instances/{entrada}.json") as file:
            data = json.load(file);
    else:
        with open("../input/inst_test.json") as file:
            data = json.load(file);
    
    if first:
        dfSolucion = pd.DataFrame();
        dfSolucion.to_excel(f"../output/CPM/ejec_output.xlsx", index=False);
        first = False;
    '''
    dfSolucion = pd.read_excel(f"../output/CPM/{entrada}_output.xlsx");
    #version = INS["version"];
    version = version;
    typePatients = INS["patients"];
    nPatients = int(INS["n_patients"]);
    nDays = int(INS["days"]);
    min_affinity = int(INS["min_affinity"]);
    nSurgeons = int(INS["surgeons"]);
    nFichas = int(INS["fichas"]);
    time_limit = int(INS["time_limit"]);

    patient_code = "0" if typePatients == "low" else ("1" if typePatients == "high" else "2");
    print(f"Versión: {version};\nPacientes: {typePatients}; Días: {nDays};\nAfinidad: {min_affinity}; Cirujanos: {nSurgeons};\nFichas: {nFichas}.");
    '''

    dfSolucion = pd.read_excel(f"../output/CPM/ejc_output.xlsx");
    #version = INS["version"];
    typePatients = data["patients"];
    nPatients = int(data["n_patients"]);
    nDays = int(data["days"]);
    min_affinity = int(data["min_affinity"]);
    nSurgeons = int(data["surgeons"]);
    nFichas = int(data["fichas"]);
    time_limit = int(data["time_limit"]);
    time_limit = 301
    bks = int(data["bks"]);
    #list_sol = [instancia[INS][0]+"_"+instancia[INS][1],"","","","","","","","","","","","","",""];
    patient_code = "0" if typePatients == "low" else ("1" if typePatients == "high" else "2");
    dict_sol = {"Instancia": [f"v{version}p{patient_code}n{nPatients}s{nSurgeons}d{nDays}"]};
    
    if typePatients == "low":
        dfPatient = pd.read_csv("../input/LowPriority.csv");
    elif typePatients == "high":
        dfPatient = pd.read_csv("../input/HighPriority.csv");
    else:
        dfPatient = pd.read_csv("../input/AllPriority.csv");
    
    dfPatient = dfPatient.iloc[:nPatients];
    
    #list_sol = [instancia[INS][0]+"_"+instancia[INS][1],"","","","","","","","","","","","","",""];
    dict_sol = {"Instancia": [f"v{version}p{patient_code}n{nPatients}s{nSurgeons}d{nDays}"]};

    # Indices
    random.seed(0);
    patient = [p for p in range(nPatients)];
    surgeon = [s for s in range(nSurgeons)];
    second = [a for a in range(nSurgeons)]
    room = [o for o in range(len(dfRoom))]
    day = [d for d in range(nDays)];
    nSlot = 16;  # Bloques de 30 minutos
    nRooms = len(room);
    slot = [t for t in range(nSlot)]
    T = nSlot//2; # División entre mañana y tarde
    
    process=[t for t in range(len(dfType))] # ?
    level_affinity = min_affinity;

    # Arcos
    arcox = [(p,o,s,a,t,d) for p in patient for o in room for s in surgeon for a in second for t in slot for d in day];
    arcoy = [(p,o,d) for p in patient for o in room for d in day];
    arcoz = [(p,s,a) for p in patient for s in surgeon for a in second];
    arcot = [p for p in patient];
    arcof = [(s,d) for s in surgeon for d in [a for a in range(-1, nDays)]];

    #Max surgeries and budget per surgeon
    M = np.zeros(nSurgeons, dtype=int);
    Pr = np.zeros(nSurgeons, dtype=int);
    for s in surgeon:
        M[s] = int(dfSurgeon.iloc[s][8]); # Máximo de cirugías
        Pr[s] = int(dfSurgeon.iloc[s][11]); # Presupuesto de fichas

    E = np.ones((nSlot, nDays)) * 1000; # ? 
    A = np.ones((nSlot, nDays)) * 1000; # ?
    B = np.ones(nPatients) * 1; # ?
    Y = np.ones(nPatients) * 1; # ?

    # Prioridades
    I = np.ones((nPatients, nDays), dtype=int);
    for p in patient:
        for d in day:
            try:
                I[(p,d)] = 1 + dfPatient.iloc[p]["espera"] * dfPatient.iloc[p]["edad"] * 0.0001/(d+1);
            except ValueError:
                print("Value Error en cálculo de prioridades.");

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
    
    aux = [list(item) for sublist in Ex for item in sublist];
    ExCP = [int(item) for sublist in aux for item in sublist];
    ExCP.extend([0 for i in range(num_ext*len(surgeon)*200)]);
    
    num_per_extra = [3,2];


    #Surgeons availability
    SDm = np.zeros((len(surgeon),len(slot),len(day)), dtype=int);
    for s in surgeon:
        for d in day:
            if dfSurgeon.iloc[s][2+d%5] == 1:
                if dfSurgeon.iloc[s][1] == 1:
                    for t in range(0,int(nSlot/2)):
                        SDm[(s,t,d)] = 1;
                if dfSurgeon.iloc[s][2] == 1:
                    for t in range(int(nSlot/2),nSlot):
                        SDm[(s,t,d)] = 1;
                        
    SDs = np.zeros((len(second),len(slot),len(day)), dtype=int);
    for a in second:
        for d in day:
            if dfSecond.iloc[a][2+d%5] == 1:
                if dfSecond.iloc[a][1] == 1:
                    for t in range(0,int(nSlot/2)):
                        SDs[(a,t,d)] = 1;
                if dfSecond.iloc[a][2] == 1:
                    for t in range(int(nSlot/2),nSlot):
                        SDs[(a,t,d)] = 1;   

    # Disponibilidad de cirujanos
    # SD = np.zeros((nSurgeons, nSurgeons, nSlot, nDays%5), dtype=int);
    # for s in surgeon:
    #     for a in second:
    #         for d in day:
    #             if dfSurgeon.iloc[s][2+d%5] == 1:
    #                 if dfSecond.iloc[a][2+d%5] == 1:
    #                     if dfSurgeon.iloc[s][1] == 1:
    #                         if dfSecond.iloc[a][1] == 1:
    #                             for t in range(nSlot//T):
    #                                 SD[(s,a,t%2,d%5)] = 1;
    #                     if dfSurgeon.iloc[s][2] == 1:
    #                         if dfSecond.iloc[a][2] == 1:
    #                             for t in range(nSlot//T, nSlot):
    #                                 SD[(s,a,t%2,d%5)] = 1;

    # Disponibilidad de paciente
    DISP = np.ones(nPatients);
    
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
        #print(dfPatient.iloc[p][21])
        for s in surgeon:
            if busquedaType(dfPatient.iloc[p]["especialidad"]) == busquedaType(dfSurgeon.iloc[s][9]):
                #print(dfPatient.iloc[p][21])
                SP[p][s] = 1;            

    # Diccionario de paciente
    dic_p = {p: [0, 0, 0, 0, 0] for p in patient};
    for p in patient:
        #dic_p[p][0] = list_patient[p] #paciente y número aleatorio asociado (entre 0 y 1)
        dic_p[p][1] = busquedaType(dfPatient.iloc[p]["especialidad"]) # ID Especialidad
        dic_p[p][2] = dfPatient.iloc[p]["nombre"]; #Nombre del paciente
        dic_p[p][3] = p; # ID
        dic_p[p][4] = dfPatient.iloc[p]["especialidad"]; # Especialidad requerida

    # Compatibilidad quirófano-paciente
    AOR = np.zeros((nPatients, nRooms, nSlot, 5));
    dicOR = {o:[] for o in room};
    j = [];
    z = [];
    ns = 0;
    for o in room:
        if o == 0:
            for d in range(5):
                for e in range(2):
                    #print(e)
                    if e == 0:                    
                        for t in range(len(slot)//2):

                            j = dfdisORA[e][d%5];
                            j = j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d%5,t,z]
                                ns+=1
                    if e==1:
                        #print('paso')
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
            #print(dicOR[nP][3])
            if str(dic_p[ns][1])==dicOR[nP][3]:
                p = dic_p[ns][3];
                o = dicOR[nP][0];
                t = dicOR[nP][2];
                d = dicOR[nP][1];
                AOR[p][o][t][d%5] = 1;
    
    #CP Affinity
    CPAffi = [];
    for s in surgeon:
        for a in second:
            CPAffi.append(dfdisAffi.iloc[a][s+1]);
    CPAffi.extend([0 for i in range(len(surgeon)*len(second)*10)]);
    
    CPAffiDia = [];
    for s in surgeon:
        for d in day:
            CPAffiDia.append(dfdisAffiDiario.iloc[d%5][s+1]);
    CPAffiDia.extend([0 for i in range(len(surgeon)*len(day)*10)]);
            
    CPAffiBloq = [];
    for s in surgeon:
        for t in range(len(slot)//8):
            CPAffiBloq.append(dfdisAffiBloque.iloc[t][s+1]);
    CPAffiBloq.extend([0 for i in range(len(slot)*20)]);

    OT = np.zeros(nPatients, dtype=int);
    for p in patient:
        OT[p] = int(dfPatient.iloc[p]["tipo"]);
    print("_" * 160);
    print('Datos Obtenidos.');

    nFichas = int((parametroFichas * 4 * nSlot * len(room) * 2 * 3 )/(len(surgeon)**0.5));
    print("Nuevas fichas:", nFichas);

    ###################################################################################################################################
    #########                                               Solución Inicial                                                  #########
    ###################################################################################################################################

    def compress(o, d, t):
        return o * nSlot * nDays + d * nSlot + t

    def decompress(val):
        o = (val) // (nSlot * nDays);
        temp = (val) % (nSlot * nDays);
        d = temp // nSlot;
        t = temp % nSlot;
        return o, d, t

    def WhichExtra(o,t,d,e):
        try:
            return int(extras[o][t][d%5][e]);
        except:
            print(f'extras: d:{d},t:{t},o:{o},e:{e}');
            print(stop)

    dictCosts = {};

    for s in surgeon:
        for a in second:
            for _ in range(nSlot * nDays * len(room)):
                o, d, t = decompress(_);
                dictCosts[(s, a, _)] = int(dfdisAffi.iloc[a][s+1] + dfdisAffiDiario.iloc[d%5][s+1] + sum(Ex[i][(s,WhichExtra(o,t//T,d,i)-1)] for i in range(num_ext)) + dfdisAffiBloque.iloc[t//T][s+1]);
    
    def generar_solucion_inicial(VERSION="A"):
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
                                        if (dfdisAffi.iloc[a][s+1] >= level_affinity*(VERSION=="B") and
                                            dfdisAffiDiario.iloc[d % 5][s+1] >= level_affinity*(VERSION=="B") and
                                            dfdisAffiBloque.iloc[t // (nSlot // 2)][s+1] >= level_affinity*(VERSION=="B")):
                                            checks = 0;
                                            for i in range(num_ext):
                                                e = int(WhichExtra(o,t//8,d%5,i));
                                                if Ex[i][(s,e-1)] >= level_affinity*(VERSION=="B"):
                                                    checks += 1;
                                            if checks >= num_ext*(VERSION=="B"):
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

        print("Solución inicial creada...")
        return asignP, dictS, dictA

    #########################################################################################################################
    #########                                      Contruccion modelo cplex                                         #########
    #########################################################################################################################
    tiempo_inicio = default_timer();
    
    from docplex.cp.model import *
    import docplex.cp.solution as Solucion
    import psutil

    def WhichExtra(o,t,d,e):
        try:
            return int(extras[o][t][d][e]);
        except:
            print(f'extras: d:{d},t:{t},o:{o},e:{e}');
            print(stop)

    mdl=CpoModel('CP Model');

    #Variables
    C = [[binary_var(name="C_"+str(p)+str(d)) for d in day] for p in patient];

    O = [integer_var(name="O_"+str(p),min=0,max=len(room)) for p in patient];
    
    M = [integer_var(name="M_"+str(p),min=0,max=len(surgeon)) for p in patient];
    
    S = [integer_var(name="S_"+str(p),min=0,max=len(second)) for p in patient];
    
    T = [integer_var(name="T_"+str(p),min=0,max=nDays*len(slot)-1) for p in patient];
    
    if version == "C":
        F = [[[integer_var(name="F_"+str(p)+str(s)+str(d), min=0) for d in range(nDays + 1)] 
            for s in range(len(surgeon)+1)] for p in patient];
    
        F_aux = [[integer_var(name="Faux_"+str(s)+str(d),min=0) for d in range(nDays + 1)] for s in range(len(surgeon)+1)];
    
        fichas = nFichas;
    
    B = nDays * len(slot);
    
    print("Cargando función objetivo...")
    
    if version == "C":
        mdl.add(maximize(sum(1000 * I[(p,d)] * C[p][d] for p in patient for d in day)
                        - sum(F_aux[s][d] * (nDays//d) for s in surgeon for d in range(1,nDays+1))));
    
    else:
        mdl.add(maximize(sum(1000 * I[(p,d)] * C[p][d] for p in patient for d in day)));

    CPAffiEx = []
    for s in surgeon:
        for d in day:
            for o in room:
                for t in range(len(slot)//8):
                    sum_puntaje = int(sum(Ex[i][(s, int(extras[o][t][d%5][i]) - 1)] for i in range(num_ext)))
                    CPAffiEx.extend([sum_puntaje])
                    
    CPAffiEx.extend([0 for i in range(len(surgeon)*len(day)*len(room)*2)]);

    #Restricciones
    
    #1: No overlap de habitaciones
    print("Cargando 1...");
    for i in range(len(patient)):
        for j in range(len(patient)):
            if i != j:
                mdl.add(mdl.if_then(mdl.logical_and(O[i] == O[j], O[i] < len(room)), T[i] != T[j]));
            
    for i in range(len(patient)):
        for j in range(len(patient)):
            if i != j:
                mdl.add(mdl.if_then(mdl.logical_and(mdl.logical_and(O[i] == O[j], O[i] < len(room)), T[i] > T[j]), T[j] + OT[j] <= T[i]));
            
    #print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
            
    #2: No overlap de cirujanos principales
    print("Cargando 2...");
    for i in range(len(patient)):
        for j in range(len(patient)):
            if i != j:
                mdl.add(mdl.if_then(mdl.logical_and(M[i] == M[j], M[i] < len(surgeon)), T[i] != T[j]));
            
    for i in range(len(patient)):
        for j in range(len(patient)):
            if i != j:
                mdl.add(mdl.if_then(mdl.logical_and(mdl.logical_and(M[i] == M[j], M[i] < len(surgeon)), T[i] > T[j]), T[j] + OT[j] <= T[i]));
            
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
            
    #3: No overlap de cirujanos secundarios
    print("Cargando 3...");
    for i in range(len(patient)):
        for j in range(len(patient)):
            if i != j:
                mdl.add(mdl.if_then(mdl.logical_and(S[i] == S[j], S[i] < len(second)), T[i] != T[j]));
            
    for i in range(len(patient)):
        for j in range(len(patient)):
            if i != j:
                mdl.add(mdl.if_then(mdl.logical_and(mdl.logical_and(S[i] == S[j], S[i] < len(second)), T[i] > T[j]), T[j] + OT[j] <= T[i]));
            
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
    
    #4: Cirujanos deben poder realizar cirugía
    
    print("Cargando 4...");
    for p in patient:
        for s in surgeon:
            if SP[(p,s)] == 0:
                mdl.add(M[p] != s);
            
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
            
    #5: Vinculación de las variables
    print("Cargando 5...");
    for p in patient:
        for d in day:
            mdl.add(mdl.if_then(C[p][d] == 1, T[p] >= d*len(slot)));
            mdl.add(mdl.if_then(C[p][d] == 1, T[p] < (d+1)*len(slot)));
            mdl.add(mdl.if_then(C[p][d] == 1, O[p] < len(room)));
            mdl.add(mdl.if_then(C[p][d] == 1, S[p] < len(second)));
            mdl.add(mdl.if_then(C[p][d] == 1, M[p] < len(surgeon)));

    for p in patient:
        mdl.add(mdl.if_then(M[p] >= len(surgeon), mdl.logical_and(O[p] >= len(room), S[p] >= len(second))));
        mdl.add(mdl.if_then(O[p] >= len(room), mdl.logical_and(M[p] >= len(surgeon), S[p] >= len(second))));
        mdl.add(mdl.if_then(S[p] >= len(second), mdl.logical_and(O[p] >= len(room), M[p] >= len(surgeon))));
        
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
    
    #6: Cirujano principal debe ser distinto a cirujano secundario
    print("Cargando 6...");
    for s in surgeon:
        for a in second:
            if COIN[(s,a)] == 1:
                for p in patient:
                    mdl.add(mdl.if_then(M[p] == s, S[p] != a));
                    
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
    
    #7: No sobrepasar límite de slots   
    print("Cargando 7...");
    for p in patient:
        for d in day:
            mdl.add(mdl.if_then(mdl.logical_and(T[p] >= d*len(slot), T[p] < len(slot)*(d+1)), 
                                T[p] + OT[p] <= len(slot)*(d+1)));
            
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
    
    #8: Compatibilidad quirófano - paciente
    print("Cargando 8...");
    for p in patient:
        for o in room:
            for t in slot:
                for d in day:
                    if AOR[(p,o,t,d%5)] == 0:
                        mdl.add(mdl.if_then(mdl.logical_and(O[p] == o, T[p] == d*len(slot) + t),
                                           O[p] == len(room)));
                
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
    
    #9: Disponibilidad cirujanos y paciente
    print("Cargando 9...");
    for p in patient:
        for s in surgeon:
            for t in slot:
                for d in day:
                    if SDm[(s,t,d%5)] == 0:
                        mdl.add(mdl.if_then(T[p] == d*len(slot) + t,
                                           M[p] != s));
                        
    for p in patient:
        for a in second:
            for t in slot:
                for d in day:
                    if SDs[(a,t,d%5)] == 0:
                        mdl.add(mdl.if_then(T[p] == d*len(slot) + t,
                                           S[p] != a));                   
                            
    for p in patient:
        if DISP[p] == 0:
            mdl.add(M[p] == len(surgeon));
            
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());

    #10: No puede usarse un bloque de la mañana y de la tarde para una operación
    for p in patient:
        mdl.add((T[p] % len(slot))//8 == ((T[p] + int(OT[p]) - 1) % len(slot))//8);
        
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
    
    #11: Penalizing Score System
    
    print("Cargando 10...");
    if version == "B":
        for s in surgeon:
            for a in second:
                if dfdisAffi.iloc[a][s+1] < level_affinity and COIN[(s,a)] == 0:
                    #print(f"Cirujano {s} ({dfSurgeon.iloc[s][0]}) no puede trabajar con secundario {a} ({dfSecond.iloc[a][0]}).")
                    for p in patient:
                        mdl.add(mdl.if_then(M[p] == s, S[p] != a));
        
        for i in range(num_ext):
            for s in surgeon:
                for o in room:
                    for t in [0,1]:
                        for d in day:
                            e = int(extras[o][t][d%5][i]);
                            if Ex[i][(s,e-1)] < level_affinity:
                                for p in patient:
                                    mdl.add(mdl.if_then(M[p] == s, mdl.logical_or(O[p] != o, 
                                                        mdl.logical_or(T[p] < d*len(slot) + t * 8, 
                                                        T[p] >= d*len(slot) + (t + 1) * 8))))
        
        for s in surgeon:
            for d in day:
                if dfdisAffiDiario.iloc[d%5][s+1] < level_affinity:
                    for p in patient:
                        mdl.add(mdl.if_then(M[p] == s, mdl.logical_or(T[p] < d*len(slot), T[p] >= (d+1)*len(slot))));
                        
        for s in surgeon:
            if dfdisAffiBloque.iloc[0][s+1] < level_affinity:
                for p in patient:
                    mdl.add(mdl.if_then(M[p] == s, T[p] % len(slot) > 7));
            if dfdisAffiBloque.iloc[1][s+1] < level_affinity:
                for p in patient:
                    mdl.add(mdl.if_then(M[p] == s, T[p] % len(slot) <= 7));
        
                        
        print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());

    elif version == "C":
        for p in patient:
            for s in surgeon:
                mdl.add(F[p][s][0] == 0);
        
        def aux_function(i_aux):
            aux = 0;
            while i_aux > 0:
                aux += i_aux*num_per_extra[i_aux-1];
                i_aux -= 1;
            return aux; 
        
        for p in patient:
            for s in surgeon:
                for d in range(1,nDays+1):
                    mdl.add(mdl.if_then(mdl.logical_and(M[p] == s, mdl.logical_and(T[p]//len(slot) >= d-1, T[p]//len(slot) < d)), 
                                        F[p][s][d] == (mdl.element(CPAffi, M[p] * len(second) + S[p])) +
                                        mdl.element(CPAffiDia, M[p] * len(day) + T[p] // len(slot)) +
                                        mdl.element(CPAffiEx, M[p]*len(day)*len(room)*2 + (T[p] // len(slot))*len(room)*2 + O[p]*2 + (T[p]%len(slot))//8) + 
                                        mdl.element(CPAffiBloq, M[p] * len(slot)//8 + (T[p] % len(slot))//8)))
                    
                    mdl.add(mdl.if_then(mdl.logical_or(M[p] != s, mdl.logical_or(T[p]//len(slot) < d-1, T[p]//len(slot) >= d)),
                                        F[p][s][d] == 0));                
                    
        for s in surgeon:
            mdl.add(F_aux[s][0] == 0);
            for d in range(1,nDays+1):
                mdl.add(F_aux[s][d] == fichas + F_aux[s][d-1] - sum(F[p][s][d] for p in patient));
        
        print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
    
    tiempo_fin = default_timer();
    tiempo_carga = str(np.around(tiempo_fin - tiempo_inicio, decimals=2));

    ########### Solución Inicial

    def validar_solucion_inicial(mdl, warmstart):
        for variable, valor in warmstart.iter_var_values():
            # Asignar temporalmente el valor propuesto al modelo
            if valor > 0:  # Solo tiene sentido validar valores no nulos
                print(f"Comprobando variable: {variable} con valor: {valor}")
                try:
                    mdl.add_constraint(variable == valor, f"check_{variable}")
                    solucion_temporal = mdl.solve()
                    #if solucion_temporal is None:
                    if mdl.get_solve_status().name == 'INFEASIBLE_SOLUTION':
                        print(f"Variable {variable} con valor {valor} viola las restricciones.")
                    # Remover la restricción temporal
                    #mdl.remove_constraint(f"check_{variable}")
                except Exception as e:
                    print(f"Error al validar la variable {variable}: {e}")

    if solucion_inicial:
        ini = generar_solucion_inicial(VERSION=version)
        fichas = [[nFichas * (d + 1) for d in range(len(day))] for s in surgeon]

        # Create an empty solution object
        warmstart = mdl.create_empty_solution()
        constant_o = len(room);
        constant_m = len(surgeon);
        constant_s = len(second);
        constant_t = nSlot * nDays;
        print("Pacientes asignados en la solución inicial:", len([item for item in ini[0] if item >= 0]));
        for p in range(len(ini[0])):
            if ini[0][p] >= 0:
                s = ini[1][ini[0][p]]
                a = ini[2][ini[0][p]]
                o, d, t = decompress(ini[0][p])

                warmstart[C[p][d]] = 1
                warmstart[O[p]] = o;
                warmstart[M[p]] = s;
                warmstart[S[p]] = a;
                warmstart[T[p]] = d*nSlot + t;

                cost = dictCosts[(s, a, ini[0][p])]
                for d_aux in range(d, nDays):
                    fichas[s][d_aux] -= cost
            else:
                warmstart[C[p][d]] = 0
                warmstart[O[p]] = constant_o;
                warmstart[M[p]] = constant_m;
                warmstart[S[p]] = constant_s;
                #warmstart[T[p]] = constant_t;

        if version == "C":
            for s in surgeon:
                for d in day:
                    warmstart[F_aux[s][d+1]] = fichas[s][d]
                    # warmstart[f[(s, d)]] = fichas[s][d]

        # validar_solucion_inicial(mdl, warmstart)
        mdl.set_starting_point(warmstart)
    
    #Resolución
    print('Restricciones Cargadas.');
    print(f'Resolviendo iterativamente para instancia: {entrada} con BKS: {bks}');

    def _calculate_gap_obj_vs_bound(obj, bound):
        if obj is None or bound is None or math.isinf(obj) or math.isinf(bound):
            return None;
        denominator = abs(obj);
        if denominator < 1e-9:
            if abs(obj - bound) < 1e-9:
                return 0.0;
            else:
                return None;
        try:
            return abs(obj - bound) / denominator;
        except Exception:
            return None;
        return None;

    def _calculate_gap_vs_bks(obj, bks_value):
        if obj is None or bks_value is None or math.isinf(obj) or math.isinf(bks_value):
            return None;
        denominator = abs(bks_value);
        if denominator < 1e-9: # If BKS is zero
            return 0.0 if abs(obj) < 1e-9 else None;
        try:
            return (bks_value - obj) / denominator;
        except Exception:
            return None;
        return None;

    chk_times = [60, 90, 120];
    if 'time_limit' in locals() and chk_times[-1] > time_limit:
        chk_times = [t for t in chk_times if t <= time_limit];
        if time_limit not in chk_times and time_limit > (chk_times[-1] if chk_times else 0) :
            chk_times.append(time_limit);

    seg_durs = [];
    prev_chk_time = 0;
    for t_chk in chk_times:
        duration = t_chk - prev_chk_time;
        if duration > 0:
            seg_durs.append(duration);
        prev_chk_time = t_chk;

    if not seg_durs and time_limit > 0:
        seg_durs = [time_limit];
        chk_times = [time_limit];
    elif not seg_durs and time_limit == 0:
        print(f"Time limit is 0 for instance {entrada}. Skipping iterative solving.");

    log_data_for_this_instance = [];
    warm_sol_for_instance = None;

    global_best_obj_for_instance = None; 
    iter_best_obj_found_at_time_for_instance = None;
    global_best_gap_vs_bks_for_instance = None;
    cumulative_branches_for_instance = 0;
    cumulative_solver_time_for_instance = 0.0;
    last_obj_found_for_instance = None;

    if 'solucion_inicial' in locals() and solucion_inicial:
        try:
            ini_sol_tuple = generar_solucion_inicial(VERSION=version);

            initial_ws = mdl.create_empty_solution();
            const_o = len(room);
            const_m = len(surgeon);
            const_s = len(second);

            asignP_from_ini, dictS_from_ini, dictA_from_ini = ini_sol_tuple;

            print("Pacientes asignados en la solución inicial:", len([item for item in asignP_from_ini if item >= 0]));
            for p_idx in patient:
                if asignP_from_ini[p_idx] >= 0:
                    o_val, d_val, t_val = decompress(asignP_from_ini[p_idx]);
                    s_val = dictS_from_ini.get(asignP_from_ini[p_idx]);
                    a_val = dictA_from_ini.get(asignP_from_ini[p_idx]);

                    if s_val is not None and a_val is not None:
                        for d_day_idx in day:
                            initial_ws[C[p_idx][d_day_idx]] = 1 if d_day_idx == d_val else 0;
                        initial_ws[O[p_idx]] = o_val;
                        initial_ws[M[p_idx]] = s_val;
                        initial_ws[S[p_idx]] = a_val;
                        initial_ws[T[p_idx]] = d_val * len(slot) + t_val;

                        if version == "C" and 'fichas' in locals() and isinstance(fichas, list) and 'dictCosts' in locals():
                            cost = dictCosts[(s_val, a_val, asignP_from_ini[p_idx])];
                    else:
                        for d_day_idx in day: initial_ws[C[p_idx][d_day_idx]] = 0;
                        initial_ws[O[p_idx]] = const_o;
                        initial_ws[M[p_idx]] = const_m;
                        initial_ws[S[p_idx]] = const_s;
                else:
                    for d_day_idx in day: initial_ws[C[p_idx][d_day_idx]] = 0;
                    initial_ws[O[p_idx]] = const_o;
                    initial_ws[M[p_idx]] = const_m;
                    initial_ws[S[p_idx]] = const_s;

            if version == "C" and 'fichas' in locals() and isinstance(fichas, list): 
                for s_idx_ws in surgeon:
                    for d_idx_ws in day: 
                        if s_idx_ws < len(fichas) and d_idx_ws < len(fichas[s_idx_ws]):
                            initial_ws[F_aux[s_idx_ws][d_idx_ws+1]] = fichas[s_idx_ws][d_idx_ws];


            mdl.set_starting_point(initial_ws);
            warm_sol_for_instance = initial_ws;
            print(f"Initial warm start solution set on the model for instance {entrada}.");
        except NameError as e:
            print(f"Skipping initial warm start for instance {entrada} due to missing variable: {e}");
        except Exception as e:
            print(f"Error during initial warm start for instance {entrada}: {e}");

    for i_seg, seg_dur_val in enumerate(seg_durs):
        print(f"--- Instance {entrada}: Starting segment {i_seg+1}/{len(seg_durs)}, duration: {seg_dur_val:.2f}s ---");

        params_seg = CpoParameters(
            TimeLimit=float(seg_dur_val),
            Workers=1,
            RelativeOptimalityTolerance=1e-5
        );

        solver_seg = mdl.create_solver(params=params_seg);
        print(f"Instance {entrada}: Solving segment {i_seg+1}...");
        seg_res = solver_seg.solve();

        seg_time_taken = seg_res.get_solve_time();
        cumulative_solver_time_for_instance += seg_time_taken;
        status_name_segment = seg_res.get_solve_status() or "UNKNOWN";
        print(f"Instance {entrada}: Segment {i_seg+1} finished. Status: {status_name_segment}, Solver time: {seg_time_taken:.2f}s");

        infos_segment = seg_res.get_solver_infos();
        branches_in_segment = infos_segment.get("NumberOfBranches", 0);
        cumulative_branches_for_instance += branches_in_segment;

        obj_current_segment = None;
        bnd_current_segment = None;

        if seg_res.is_solution():
            obj_current_segment = seg_res.get_objective_value();
            bnd_current_segment = seg_res.get_objective_bound();
            print(f"Instance {entrada}: Solution found in segment: Obj={obj_current_segment}, Bound={bnd_current_segment}");

            if obj_current_segment is not None:
                last_obj_found_for_instance = obj_current_segment;
                if global_best_obj_for_instance is None or obj_current_segment > global_best_obj_for_instance:
                    global_best_obj_for_instance = obj_current_segment;
                    iter_best_obj_found_at_time_for_instance = chk_times[i_seg];

            warm_sol_for_instance = seg_res.solution;
            mdl.set_starting_point(warm_sol_for_instance);
        else:
            obj_current_segment = last_obj_found_for_instance;
            bnd_current_segment = seg_res.get_objective_bound();
            print(f"Instance {entrada}: No new solution in this segment. Using last known Obj={obj_current_segment}, Current Bound={bnd_current_segment}");

        gap_vs_bound_c = _calculate_gap_obj_vs_bound(obj_current_segment, bnd_current_segment);
        gap_vs_bks_c = _calculate_gap_vs_bks(obj_current_segment, bks);

        if gap_vs_bks_c is not None:
            if global_best_gap_vs_bks_for_instance is None or \
            (gap_vs_bks_c >= 0 and gap_vs_bks_c < global_best_gap_vs_bks_for_instance) or \
            (gap_vs_bks_c < 0 and (global_best_gap_vs_bks_for_instance < 0 and gap_vs_bks_c > global_best_gap_vs_bks_for_instance)):
                global_best_gap_vs_bks_for_instance = gap_vs_bks_c;

        target_cumulative_time_log = chk_times[i_seg];
        row = {
            'instance': entrada,
            'time': target_cumulative_time_log,
            'objective': obj_current_segment,
            'gap': gap_vs_bks_c,
            'best_gap': global_best_gap_vs_bks_for_instance,
            'best_obj': global_best_obj_for_instance
        };
        log_data_for_this_instance.append(row);

        if i_seg < len(seg_durs) - 1:
            current_best_to_report = global_best_obj_for_instance if global_best_obj_for_instance is not None else "N/A";
            print(f"Instance {entrada}: Preparing for next segment. Current best objective for warm start: {current_best_to_report}");
        else:
            print(f"Instance {entrada}: All segments completed.");
    if log_data_for_this_instance:
        df_instance_checkpoints = pd.DataFrame(log_data_for_this_instance);

        write_header = not global_checkpoint_csv_path.exists() or global_checkpoint_csv_path.stat().st_size == 0;
        df_instance_checkpoints.to_csv(global_checkpoint_csv_path, mode='a', header=write_header, index=False);
        print(f"Checkpoint data for instance {entrada} appended to {global_checkpoint_csv_path}");

    print(f"Total solver time across all segments for instance {entrada}: {cumulative_solver_time_for_instance:.2f} seconds.");

    solucion = seg_res if 'seg_res' in locals() and 'i_seg' in locals() else mdl.create_empty_solution(); # Fallback if loop didn't run
    if not (seg_res if 'seg_res' in locals() and 'i_seg' in locals() else None): # if seg_res is not defined
        print(f"Warning: seg_res not available for instance {entrada} after iterative solving. Solucion object might be empty.");

    #mdl.add_progress_listener(listener)
    #solucion = mdl.solve(TimeLimit=time_limit,Workers=1,RelativeOptimalityTolerance=0.00001);
        
    infos = solucion.get_solver_infos();
    #info_memoria = (infos['MemoryUsage'],infos['PeakMemoryUsage']);  # Display memory usage of the CP engine
    #list_sol[14] = str(np.around(info_memoria[0]/10**6, decimals=2));
    memoria_usada = np.around(psutil.Process().memory_info().rss / 10**6, decimals=2);
    print("Memoria usada:", memoria_usada, "megabytes");
    dict_sol["Memoria"] = [memoria_usada];

    #print('status:',mdl.get_solve_status());
    if solucion.is_solution():
        if solucion.is_solution_optimal():
            #list_sol[5] = "Opt";
            dict_sol["Status"] = ["Opt"];
        else:
            #list_sol[5] = "Fact";
            dict_sol["Status"] = ["Fact"];
    else:
        dict_sol["Status"] = ["Infact"];
        dict_sol["Valor FO"] = [0];
        dict_sol["Best Bound"] = [0];
        dict_sol["Rel GAP"] = [0];
        time_fin = time.time()
        time_ejecucion = time_fin - time_inicio
        print("Tiempo ejecucion:",time_ejecucion);
        dict_sol["Tiempo Carga"] = [tiempo_carga];
        dict_sol["Tiempo Ejec"] = [np.around(time_ejecucion, decimals=2)];
        dict_sol["Pacientes Atend"] = [0];
        dict_sol["Prioridad"] = [0];
        dict_sol["Avg Fichas"] = [0];
        dict_sol["Std Fichas"] = [0];
        dict_sol["Avg Cirug"] = [0];
        dict_sol["Std Cirug"] = [0];
        dict_sol["Avg Ratio"] = [0];
        dict_sol["Std Ratio"] = [0];
        dict_sol["Ocupación"] = [0];
        dfAux = pd.DataFrame(dict_sol);
        dfSolucion = pd.concat([dfSolucion, dfAux]);
        dfSolucion.to_excel(f"./Output/CPM/{entrada}_output.xlsx", index=False);
        continue;

    print('1:',solucion.get_objective_value())
    #list_sol[1] = str(solucion.get_objective_value());
    dict_sol["Valor FO"] = [solucion.get_objective_value()];

    #print('Best bound:',solucion.solve_details.best_bound)
    #list_sol[2] = str(solucion.get_objective_bound());
    dict_sol["Best Bound"] = [np.around(solucion.get_objective_bound(), decimals=2)];

    #list_sol[6] = str(np.around(solucion.get_objective_gap(),decimals=2));
    dict_sol["Rel GAP"] = [np.around(solucion.get_objective_gap(), decimals=2)];

    #print('Time:',solucion.solve_details.time)

    time_fin = default_timer()
    time_ejecucion = time_fin - time_inicio
    print("Tiempo ejecucion:",time_ejecucion);
    print("Objetivo:",solucion.get_objective_value());
    aux = np.around(time_ejecucion,decimals=2);
    #list_sol[3] = str(tiempo_carga);
    dict_sol["Tiempo Carga"] = [tiempo_carga];
    #list_sol[4] = str(aux);
    dict_sol["Tiempo Ejec"] = [np.around(time_ejecucion, decimals=2)];
    #%%
    pacientes_atend = [];
    fichas_ocup = [0 for i in range(len(surgeon))];
    cirujanos_atend_num = [0 for i in range(len(surgeon))];
    prioridad = 0;
    ocupacion = 0;
    ratios = [0 for i in range(len(surgeon))];
    #PROG={e:[None, None, None, None, None] for e in range(100)}
    PROG = {e:[None, None, None, None, None, None, None, None, None, None] for e in range(nSlot * nDays * len(room) + nDays*(len(room) + 1) + 20)};
    c=0
    lista_val = [];
    lista_p = [];
    for p in patient:
        val = solucion.get_solution().get_value(T[p]);
        m = solucion.get_solution().get_value(M[p])
        o = solucion.get_solution().get_value(O[p])
        if m < len(surgeon):
            lista_val.append(val);
            #print(f"Sol var O[{p}] = {solucion.get_solution().get_value(O[p])}");
            #print(f"Sol var T[{p}] = {solucion.get_solution().get_value(T[p])}");
            #print(f"Sol var M[{p}] = {solucion.get_solution().get_value(M[p])}");
            #print(f"Sol var S[{p}] = {solucion.get_solution().get_value(S[p])}");
            #print(CPAffiEx[m*len(day)*len(room)*2 + (val//len(slot))*len(room)*2 + o*2 + (val%len(slot))//8]);
            lista_p.append(p);
        else:
            #print(f"Sol var O[{p}] = {solucion.get_solution().get_value(O[p])}");
            #print(f"Sol var T[{p}] = {solucion.get_solution().get_value(T[p])}");
            #print(f"Sol var M[{p}] = {solucion.get_solution().get_value(M[p])}");
            #print(f"Sol var S[{p}] = {solucion.get_solution().get_value(S[p])}");
            #print(m*len(day)*len(room)*2 + (val//len(slot))*len(room)*2 + o*2 + (val%len(slot))//8, len(CPAffiEx))
            #print(CPAffiEx[m*len(day)*len(room)*2 + (val//len(slot))*len(room)*2 + o*2 + (val%len(slot))//8]);
            pass;
    lista_p = [x for _,x in sorted(zip(lista_val,lista_p))];
    
    print("Lista p:", lista_p);

    fichas_ocup = [0 for i in range(len(surgeon))];
    
    for d in day:
        PROG[c][0] = 'DIA ' + str(d);
        c += 1;
        for o in room:
            PROG[c][0] = 'PABELLON ' + str(o);
            c += 1;
            for p in lista_p:
                dia = solucion.get_solution().get_value(T[p]) // len(slot);
                bloque = solucion.get_solution().get_value(T[p]) % len(slot);
                hab = solucion.get_solution().get_value(O[p]);
                #print("dia:",dia)
                #print("hab:",hab)
                if dia == d and hab == o:
                    for j in range(int(OT[p])):
                        #print("p:",p,"O_p:",solucion.get_solution().get_value(O[p]))
                        tiempo = solucion.get_solution().get_value(T[p]);
                        surg = solucion.get_solution().get_value(M[p]);
                        sec = solucion.get_solution().get_value(S[p]);

                        PROG[c][0]= 'BLOQUE ' + str(tiempo % len(slot) + j);
                        PROG[c][1]= dfPatient.iloc[p][2];
                        if dfPatient.iloc[p]["id"] not in pacientes_atend:
                            pacientes_atend.append(dfPatient.iloc[p]["id"]);
                            fichas_ocup[surg] += dictCosts[surg, sec, compress(o, d, t)]
                            cirujanos_atend_num[surg] += 1;
                            prioridad += I[(p,d)];
                            ocupacion += int(OT[p]);
                        PROG[c][2]= dfSurgeon.iloc[surg][0];
                        PROG[c][3]= dfSecond.iloc[sec][0];
                        PROG[c][4]= dfPatient.iloc[p]["especialidad"];
                        PROG[c][5] = dfdisAffi.iloc[sec][surg+1];
                        aux = 0;
                        for i in range(num_ext):
                            aux += int(Ex[i][(surg, WhichExtra(o, (tiempo % nSlot) // (nSlot // 2), d%5, i) - 1)]);
                        PROG[c][6] = aux;
                        PROG[c][7] = dfdisAffiDiario.iloc[d%5][surg+1];
                        PROG[c][8] = dfdisAffiBloque.iloc[(tiempo % nSlot)//8][surg+1];
                        PROG[c][9] = int(PROG[c][5]) + int(PROG[c][6]) + int(PROG[c][7]) + int(PROG[c][8]);
                        c += 1;

    dfPROG = pd.DataFrame({'BLOQUE':[PROG[c][0] for c in PROG],
        'PACIENTE':[PROG[c][1] for c in PROG],
        '1ER CIRUJANO':[PROG[c][2] for c in PROG],
        '2DO CIRUJANO':[PROG[c][3] for c in PROG],
        'TIPO PROC':[PROG[c][4] for c in PROG],
        'FICHAS CIR':[PROG[c][5] for c in PROG],
        'FICHAS EXT':[PROG[c][6] for c in PROG],
        'FICHAS DIA':[PROG[c][7] for c in PROG],
        'FICHAS BLO':[PROG[c][8] for c in PROG],
        'TOTAL FICHAS':[PROG[c][9] for c in PROG]});
    
    #dfSTATS = pd.DataFrame([[dfSurgeon.iloc[s][0] + ": ",str(sum(solucion.get_solution().get_value(F[p][s][nDays]) for p in patient))] for s in surgeon]);
    #print(dfSTATS);
    
    #writer = ExcelWriter(carpeta + "Resultados/Constraint Programming/CPM C/" + "CPM_C_" +instancia[INS][0]+"_"+instancia[INS][1]+"_"+"x"+"_"+instancia[INS][3]+".xlsx");
    dfPROG.to_excel(f"./Output/CPM/{entrada}_Schedule_v{version}p{patient_code}n{nPatients}s{nSurgeons}d{nDays}.xlsx", index=False);
    #dfPROG.to_excel(writer, 'PROGRAMACION QX', index=False);
    #dfSTATS.to_excel(writer, 'STATS CIRUJANOS', index=False);
    #writer.save();
    print("Pacientes atendidos:",len(pacientes_atend));
    #list_sol[7] = str(len(pacientes_atend));
    dict_sol["Pacientes Atend"] = [len(pacientes_atend)];

    dict_sol["Prioridad"] = [prioridad];

    #list_sol[8] = str(np.mean(fichas_ocup));
    dict_sol["Avg Fichas"] = [np.around(np.mean([i for i in fichas_ocup if i != 0]), decimals=2)];

    #list_sol[9] = str(np.std(fichas_ocup));
    dict_sol["Std Fichas"] = [np.around(np.std([i for i in fichas_ocup if i != 0]), decimals=2)];

    #list_sol[10] = str(np.mean(cirujanos_atend_num));
    dict_sol["Avg Cirug"] = [np.around(np.mean([i for i in cirujanos_atend_num if i != 0]), decimals=2)];
    
    #list_sol[11] = str(np.around(np.std(cirujanos_atend_num),decimals=2));
    dict_sol["Std Cirug"] = [np.around(np.std([i for i in cirujanos_atend_num if i != 0]), decimals=2)];

    for i in range(nSurgeons):
        if cirujanos_atend_num[i] != 0:
            ratios[i] = np.around(fichas_ocup[i]/cirujanos_atend_num[i], decimals=2);
            
    #list_sol[12] = str(round(np.mean(ratios),1));
    dict_sol["Avg Ratio"] = [np.around(np.mean([i for i in ratios if i != 0]), decimals=2)];
    
    #list_sol[13] = str(round(np.std(ratios),1)) + "\n";
    dict_sol["Std Ratio"] = [np.around(np.std([i for i in ratios if i != 0]), decimals=2)];

    dict_sol["Ocupación"] = [np.around(ocupacion/(nSlot*nDays*len(room)), decimals=2)];
    
    dfAux = pd.DataFrame(dict_sol);
    dfSolucion = pd.concat([dfSolucion, dfAux]);
            
    dfSolucion.to_excel(f"./Output/CPM/{entrada}_output.xlsx", index=False);
print('Fin del programa;');