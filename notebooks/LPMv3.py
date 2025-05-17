testing = False;
parametroFichas = 0.11;
entrada = "instance1";
version = "C";
solucion_inicial = True;
print(f"Configurado con testing = {testing}. Versión: {version}. Entrada: {entrada}. Solución inicial: {solucion_inicial}.");

import warnings
import json
import pandas as pd
import random
import numpy as np
import time
from pathlib import Path

warnings.filterwarnings("ignore");
    
#config = data["configurations"];

global_ilp_checkpoint_csv_path = Path("../output/LPM/LPM_checkpoints.csv");
global_ilp_checkpoint_csv_path.parent.mkdir(parents=True, exist_ok=True);
if global_ilp_checkpoint_csv_path.exists():
    global_ilp_checkpoint_csv_path.unlink();
                    
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

# Rankings para extras
dfRankExtra = [];
for i in range(num_ext):
    aux = pd.read_excel("../input/RANKING.xlsx", sheet_name='Extra'+str(i+1), converters = {'n°':int}, index_col=[0]);
    extra.append(len(aux));
    dfRankExtra.append(aux);

#-------------------------------------------------------------------------------------------------------------------------------#     
########################################################### MAIN LOOP ###########################################################
#-------------------------------------------------------------------------------------------------------------------------------#  

first = True;
#for INS in config:
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
        dfSolucion.to_excel(f"../output/LPM/ejec_output.xlsx", index=False);
        first = False;
    '''
    dfSolucion = pd.read_excel(f"../output/LPM/{entrada}_output.xlsx");
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
    print(f"Versión: {version};\nPacientes: {typePatients}; Número de Pacientes: {nPatients};\nDías: {nDays};\nAfinidad: {min_affinity}; Cirujanos: {nSurgeons};\nFichas: {nFichas}.");

    '''

    dfSolucion = pd.read_excel(f"../output/LPM/ejec_output.xlsx");
    #version = INS["version"];
    typePatients = data["patients"];
    nPatients = int(data["n_patients"]);
    nDays = int(data["days"]);
    min_affinity = int(data["min_affinity"]);
    nSurgeons = int(data["surgeons"]);
    nFichas = int(data["fichas"]);
    time_limit = int(data["time_limit"]);
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

    OT = np.zeros(nPatients, dtype=int);
    for p in patient:
        OT[p] = int(dfPatient.iloc[p]["tipo"]);
    #print("_" * 160);

    nFichas = int((parametroFichas * 4 * nSlot * len(room) * 2 * 3 )/(len(surgeon)**0.5));
    #print("Nuevas fichas:", nFichas);
    #print('Datos Obtenidos.');

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

        #print("Solución inicial creada...")
        return asignP, dictS, dictA

    ###################################################################################################################################
    #########                                           Contruccion modelo cplex                                              #########
    ###################################################################################################################################
    
    inicio_carga = time.time();
    import psutil
    import os
    from docplex.mp.model import Model
    import docplex.mp.solution as Solucion
    from docplex.mp.progress import ProgressListener, ProgressClock

    def WhichExtra(o,t,d,e):
        try:
            return int(extras[o][t][d%5][e]);
        except:
            print(f'extras: d:{d%5},t:{t},o:{o},e:{e}');
            #print(stop)

    #mdl=Model('MIP Model',ignore_names=True,checker='off');
    mdl = Model('MIP Model',checker='off');

    # Variables
    x = mdl.binary_var_dict(arcox, name='x');
    y = mdl.binary_var_dict(arcoy, name='y');
    z = mdl.binary_var_dict(arcoz, name='z');
    f = mdl.integer_var_dict(arcof, lb=0, name='f');
    ts = mdl.integer_var_dict(arcot, lb=0, ub=nSlot-1, name='ts');
    te = mdl.integer_var_dict(arcot, lb=0, ub=nSlot-1, name='te');

    def decompress(val):
        o = (val) // (nSlot * nDays);
        temp = (val) % (nSlot * nDays);
        d = temp // nSlot;
        t = temp % nSlot;
        return o, d, t
    
    fichas = nFichas;
    def multiplicador(dia):
        return nDays//(dia+1);
    
    print("Cargando función objetivo...")
    if version == "C":
        mdl.maximize(mdl.sum(1000*I[(p,d)]*y[(p,o,d)] for p in patient for o in room for d in day) 
                     - mdl.sum(f[(s,d)] * multiplicador(d) for s in surgeon for d in day));
    else:
        mdl.maximize(mdl.sum(1000*I[(p,d)]*y[(p,o,d)] for p in patient for o in room for d in day));       
    
    # Restricciones
    '''
    print("Cargando (2)...");
    for s in surgeon:
        for d in day:
            mdl.add_constraint(f[(s,d)] == fichas + f[(s,d-1)] - 
                               sum((dfdisAffi.iloc[a][s+1] + dfdisAffiDiario.iloc[d][s+1]
                                    + sum(Ex[i][(s,WhichExtra(o,t//4,d,i)-1)] for i in range(num_ext))
                                    + dfdisAffiBloque.iloc[t//4][s+1]) * (1/OT[p]) * x[(p,o,s,a,t,d)] 
                                   for p in patient for o in room for a in second for t in slot));

    for s in surgeon:
        mdl.add_constraint(f[(s,-1)] == 0);
    '''
    array = dfdisAffi.values
    lala = dfdisAffiDiario.values
    lele = dfdisAffiBloque.values
    
    if version == "B":
        print("Cargando (2-B)...");
        for p in patient:
            for s in surgeon:
                for a in second:
                     mdl.add_constraint(z[(p,s,a)]*dfdisAffi.iloc[a][s+1] >= level_affinity*z[(p,s,a)]);

        for p in patient:
            for s in surgeon:
                for t in slot:
                    for d in day:
                        mdl.add_constraint(mdl.scal_prod(mdl.sum_vars(x[(p,o,s,a,t,d)] for o in room for a in second),
                                                         dfdisAffiDiario.iloc[d%5][s+1]) >= level_affinity*mdl.sum_vars(x[(p,o,s,a,t,d)] for o in room for a in second));

        for p in patient:
            for s in surgeon:
                for t in slot:
                    mdl.add_constraint(mdl.scal_prod(mdl.sum_vars(x[(p,o,s,a,t,d)] for o in room for a in second for d in day),
                                                     dfdisAffiBloque.iloc[t//(nSlot//2)][s+1]) >= level_affinity*mdl.sum_vars(x[(p,o,s,a,t,d)] for o in room for a in second for d in day));
    
        for i in range(num_ext):
            for o in room:
                for s in surgeon:
                    for t in slot:
                        for d in day:
                            e = int(WhichExtra(o,t//8,d%5,i));
                            try:
                                mdl.add_constraint(mdl.scal_prod(mdl.sum_vars(x[(p,o,s,a,t,d)] for p in patient for a in second),Ex[i][(s,e-1)]) >= level_affinity*mdl.sum_vars(x[(p,o,s,a,t,d)] for p in patient for a in second));
                            except:
                                print(f'i:{i}, o:{o}, s:{s},t:{t},d:{d},e:{e}');
    

    elif version == "C":

        dictCosts = {};
        for s in surgeon:
            for a in second:
                for _ in range(nSlot * nDays * len(room)):
                    o, d, t = decompress(_);
                    dictCosts[(s, a, _)] = int(dfdisAffi.iloc[a][s+1] + dfdisAffiDiario.iloc[d%5][s+1] + sum(Ex[i][(s,WhichExtra(o,t//T,d,i)-1)] for i in range(num_ext)) + dfdisAffiBloque.iloc[t//T][s+1]);
        '''        
        print("Cargando (2-C)...");
        for s in surgeon:
            for d in day:
                mdl.add_constraint(f[(s,d)] == fichas + f[(s,d-1)] - 
                                   #mdl.sum((dfdisAffi.iloc[a][s+1] + dfdisAffiDiario.iloc[d][s+1]
                                   mdl.sum((array[a][s] + lala[d%5][s]
                                       + mdl.sum(Ex[i][(s,WhichExtra(o,t//T,d,i)-1)] for i in range(num_ext))
                                   #    + dfdisAffiBloque.iloc[t//4][s+1]) * (1/OT[p]) * x[(p,o,s,a,t,d)]
                                       + lele[t//T][s]) * (1/OT[p]) * x[(p,o,s,a,t,d)]  
                                       for p in patient for o in room for a in second for t in slot));
        '''
        for s in surgeon:
            mdl.add_constraint(f[(s,-1)] == 0);
            for d in day:
                mdl.add_constraint(f[(s,d)] == fichas + f[(s,d-1)] - mdl.sum(dictCosts[(s, a, compress(o, d, t))] * (1/OT[p]) * x[(p,o,s,a,t,d)] 
                                                                               for p in patient for o in room for a in second for t in slot));
        
    print("Cargando (3)...");
    for p in patient:
        for t in slot:
            mdl.add_constraint(mdl.sum_vars(x[(p,o,s,a,t,d)] for o in room for s in surgeon for a in second for d in day) <= 1);
    
    
    print("Cargando (4)...");
    for o in room:
        for t in slot:
            for d in day:
                mdl.add_constraint(mdl.sum_vars(x[(p,o,s,a,t,d)] for p in patient for s in surgeon for a in second) <= 1);
    
    print("Cargando (5)...");
    for s in surgeon:
        for t in slot:
            for d in day:
                mdl.add_constraint(mdl.sum_vars(x[(p,o,s,a,t,d)] for p in patient for o in room for a in second) <= 1);
    
    print("Cargando (6)...");
    for a in second:
        for t in slot:
            for d in day:
                mdl.add_constraint(mdl.sum_vars(x[(p,o,s,a,t,d)] for p in patient for o in room for s in surgeon) <= 1);

    
    print("Cargando (7)...");
    for p in patient:
        for s in surgeon:
            for a in second:
                for t in slot:
                    for d in day:
                        mdl.add_constraint(mdl.sum((mdl.scal_prod((x[(p,o,s,a,t,d)] for o in room), COIN[(s,a)]))) <= 0);
                        
    print("Cargando (8)...");
    for p in patient:
        mdl.add_constraint(mdl.sum_vars(y[(p,o,d)] for o in room for d in day) <= 1);
    
    print("Cargando (9)...");
    for p in patient:
        mdl.add_constraint(mdl.sum_vars(z[(p,s,a)] for s in surgeon for a in second) <= 1);
     
       
    #print("Cargando (10)...");
    #(4.10)
    #for p in patient:
    #    for o in room:
    #        for d in day:
    #            mdl.add_constraint(mdl.sum_vars(x[(p,o,s,a,t,d)] for s in surgeon for a in second for t in slot)
    #                               <=len(slot)*y[(p,o,d)])         

    
    print("Cargando (11)...");
    #(4.11*)
    for p in patient:
         mdl.add_constraint(mdl.sum_vars(y[(p,o,d)] for o in room for d in day) <= mdl.sum_vars(z[(p,s,a)] for s in surgeon for a in second));
    
    
    # print("Cargando (12)...");
    # for p in patient:
    #     for s in surgeon:
    #         for a in second:
    #             for t in slot:
    #                 for d in day:
    #                     mdl.add_constraint(mdl.sum_vars(x[(p,o,s,a,t,d)] for o in room) <= SD[(s,a,t,d%5)] * z[(p,s,a)]);
    
    
    print("Cargando (13)...");
    for p in patient:
        mdl.add_constraint(mdl.sum_vars(x[(p,o,s,a,t,d)] for o in room for s in surgeon for a in second for t in slot for d in day) <= nSlot * DISP[p]);
        
    print("Cargando (14)...");
    for p in patient:
        for s in surgeon:
            for d in day:
                mdl.add_constraint(mdl.sum_vars(x[(p,o,s,a,t,d)] for o in room for a in second for t in slot) <= nSlot * SP[(p,s)]);

    print("Cargando (15)...");
    for p in patient:
        for o in room:
            for t in slot:
                for d in day:
                    mdl.add_constraint(mdl.sum(x[(p,o,s,a,t,d)] for s in surgeon for a in second) <= AOR[(p,o,t,d%5)]);
    
    
    print("Cargando (16)...");
    for p in patient:
        for s in surgeon:
            for a in second:
                mdl.add_constraint(mdl.sum_vars(x[(p,o,s,a,t,d)] for o in room for t in slot for d in day) >= (OT[p]) * z[(p,s,a)]);

    print("Cargando (17)...");
    for t in slot:
        for p in patient:
            mdl.add_constraint(ts[(p)] <= t * mdl.sum_vars(x[(p,o,s,a,t,d)] for o in room for s in surgeon for a in second for d in day)
                              + nSlot * (1 - mdl.sum_vars(x[(p,o,s,a,t,d)] for o in room for s in surgeon for a in second for d in day)));
    
    print("Cargando (18)...");
    for t in slot:
        for p in patient:
            mdl.add_constraint(te[(p)] >= t * mdl.sum_vars(x[(p,o,s,a,t,d)] for o in room for s in surgeon for a in second for d in day));
    
    print("Cargando (19)...");
    for p in patient:
        mdl.add_constraint(ts[(p)] == te[(p)] - OT[p] + 1);
        
    print("Cargando (20)..."); #No se puede usar un bloque de la mañana y de la tarde para una misma operación
    for p in patient:
        for d in day:
            mdl.add_constraint(mdl.sum_vars(x[(p,o,s,a,T-1,d)] for o in room for s in surgeon for a in second) <=
                              1 - mdl.sum_vars(x[(p,o,s,a,T,d)] for o in room for s in surgeon for a in second));
           
    print("Cargando (21)..."); #Evitar asignar a rooms diferentes cuando la operación dura más de un bloque
    for p in patient:
        for d in day:
            for o in room:
                mdl.add_constraint(mdl.sum(x[(p, o, s, a, t, d)] for t in slot for s in surgeon for a in second) <= OT[p] * y[(p, o, d)]);

    for p in patient:
        mdl.add_constraint(mdl.sum(y[(p, o, d)] for o in room for d in day) <= 1);
    
    print("Cargando (22)...");
    for k in surgeon:               
        for t in slot:
            for d in day:
                mdl.add_constraint(
                    mdl.sum_vars(x[(p,o,k,a,t,d)] for p in patient for o in room for a in second)   # k as PRIMARY
                + mdl.sum_vars(x[(p,o,s,k,t,d)] for p in patient for o in room for s in surgeon)  # k as SECONDARY
                <= 1);
    
    
    fin_carga = time.time();
    tiempo_carga = str(np.around(fin_carga - inicio_carga, decimals=2));
    #print(mdl.export_to_string())
    print('Restricciones Cargadas.\nComenzando resolución...\n');
    time_inicio = time.time();
    #mdl.parameters.timelimit = time_limit;
    #mdl.parameters.timelimit = 150;
    mdl.parameters.threads = 1;

    ########### Solución Inicial
    import math

    def validar_solucion_inicial(mdl, warmstart):
        for variable, valor in warmstart.iter_var_values():
            if valor > 0:
                print(f"Comprobando variable: {variable} con valor: {valor}")
                try:
                    mdl.add_constraint(variable == valor, f"check_{variable}")
                    solucion_temporal = mdl.solve()
                    if mdl.get_solve_status().name == 'INFEASIBLE_SOLUTION':
                        print(f"Variable {variable} con valor {valor} viola las restricciones.")
                    #mdl.remove_constraint(f"check_{variable}")
                except Exception as e:
                    print(f"Error al validar la variable {variable}: {e}")

    if solucion_inicial == True:
        ini = generar_solucion_inicial(VERSION=version);
        fichas = [[nFichas * (d+1) for d in range(len(day))] for s in surgeon];
        warmstart = mdl.new_solution();
        print("Pacientes asignados en la solución inicial:", len([item for item in ini[0] if item >= 0]));
        for p in range(len(ini[0])):
            if ini[0][p] >= 0:
                s = ini[1][ini[0][p]];
                a = ini[2][ini[0][p]];
                o, d, t = decompress(ini[0][p]);
                warmstart.add_var_value(x[(p,o,s,a,t,d)], 1);
                warmstart.add_var_value(y[(p,o,d)], 1);
                warmstart.add_var_value(z[(p,s,a)], 1);
                warmstart.add_var_value(ts[(p)], t);
                warmstart.add_var_value(te[(p)], t + int(OT[p]) - 1);
                cost  = dictCosts[(s, a, ini[0][p])];
                for d_aux in range(d, nDays):
                    fichas[s][d_aux] -= cost;
        if version == "C":
            for s in surgeon:
                for d in day:
                    warmstart.add_var_value(f[(s,d)], fichas[s][d]);

        #validar_solucion_inicial(mdl, warmstart);
        mdl.add_mip_start(warmstart);

    def _calculate_gap_vs_bks_ilp(obj, bks_value):
        if obj is None or bks_value is None or math.isinf(obj) or math.isinf(bks_value):
            return None;
        denominator = abs(bks_value);
        if denominator < 1e-9:
            return 0.0 if abs(obj) < 1e-9 else None;
        try:
            return (bks_value - obj) / denominator;
        except Exception:
            return None;
        return None;

    chk_times = [30, 45, 60];
    '''
    if 'time_limit' in locals() and isinstance(time_limit, (int, float)):
        actual_time_limit = float(time_limit);
        chk_times = [t for t in chk_times if t <= actual_time_limit];
        if actual_time_limit not in chk_times and (not chk_times or actual_time_limit > chk_times[-1]):
            chk_times.append(actual_time_limit);
        chk_times = sorted(list(set(chk_times)));
    else:
        print(f"Warning: 'time_limit' not properly defined for instance {entrada}. Using default checkpoints.");
        actual_time_limit = chk_times[-1] if chk_times else 300;
    '''

    actual_time_limit = chk_times[-1] + 5;

    seg_durs = [];
    prev_chk_time_for_seg_calc = 0;
    for t_chk_for_seg_calc in chk_times:
        duration = t_chk_for_seg_calc - prev_chk_time_for_seg_calc;
        if duration > 1e-6:
            seg_durs.append(duration);
        prev_chk_time_for_seg_calc = t_chk_for_seg_calc;

    if not seg_durs and actual_time_limit > 1e-6:
        seg_durs = [actual_time_limit];
        if not chk_times or chk_times[-1] != actual_time_limit : chk_times = [actual_time_limit];

    log_data_for_this_ilp_instance = [];
    warm_start_sol_for_ilp_instance = None;

    global_best_obj_for_ilp_instance = None;
    global_best_gap_vs_bks_for_ilp_instance = None;
    cumulative_solver_time_for_ilp_instance = 0.0;
    last_obj_found_for_ilp_instance = None;

    final_sol_object_from_iterative_solve = None;
    final_solve_details_for_solucion = None;
    final_status_name_for_solucion = None;

    mdl.parameters.threads.set(1);

    if not seg_durs:
        print(f"Instance {entrada}: No segments to run (time limit {actual_time_limit}s might be too short or zero).");
    else:
        for i_seg, seg_dur_val in enumerate(seg_durs):
            remaining_time_for_solve = actual_time_limit - cumulative_solver_time_for_ilp_instance;
            current_segment_timelimit = min(seg_dur_val, remaining_time_for_solve);

            if current_segment_timelimit <= 1e-6:
                print(f"Instance {entrada}: Negligible time left ({current_segment_timelimit:.2f}s) for segment {i_seg+1}. Skipping.");
                break;

            print(f"--- ILP Instance {entrada}: Segment {i_seg+1}/{len(seg_durs)}, requested duration: {current_segment_timelimit:.2f}s ---");
            mdl.parameters.timelimit.set(float(current_segment_timelimit));

            if warm_start_sol_for_ilp_instance:
                try:
                    mdl.add_mip_start(warm_start_sol_for_ilp_instance);
                    print(f"Instance {entrada}: MIP start applied for segment {i_seg+1}.");
                except Exception as e_mip:
                    print(f"Instance {entrada}: Error applying MIP start - {e_mip}");

            sol_segment = None;
            obj_current_segment_for_log = last_obj_found_for_ilp_instance;
            current_segment_status_name = "SolveNotAttemptedOrFailedEarly";
            segment_actual_solve_time = 0.0;

            try:
                sol_segment = mdl.solve(log_output=False);

                segment_actual_solve_time = mdl.solve_details.time if mdl.solve_details else 0.0;
                current_segment_status_name = mdl.get_solve_status().name if mdl.get_solve_status() else "StatusUnavailable";

                if sol_segment:
                    if current_segment_status_name in ['OPTIMAL_SOLUTION', 'FEASIBLE_SOLUTION']:
                        obj_current_segment_for_log = sol_segment.objective_value;
                        last_obj_found_for_ilp_instance = obj_current_segment_for_log;

                        final_sol_object_from_iterative_solve = sol_segment;
                        final_solve_details_for_solucion = mdl.solve_details;
                        final_status_name_for_solucion = current_segment_status_name;

                        new_mip_start_solution = mdl.new_solution();
                        populated_vars_count = 0;
                        for var, val in sol_segment.iter_var_values():
                            new_mip_start_solution.add_var_value(var, val);
                            populated_vars_count += 1;

                        if populated_vars_count > 0:
                            warm_start_sol_for_ilp_instance = new_mip_start_solution;
                            print(f"Instance {entrada}: Feasible/Optimal solution. Obj={obj_current_segment_for_log:.2f}. Status: {current_segment_status_name}. MIP start updated.");
                        else:
                            print(f"Instance {entrada}: Feasible/Optimal status ({current_segment_status_name}) but no vars in iter_var_values. Obj={obj_current_segment_for_log:.2f}. MIP start not updated.");

                        if global_best_obj_for_ilp_instance is None or \
                        (obj_current_segment_for_log is not None and obj_current_segment_for_log > global_best_obj_for_ilp_instance):
                            global_best_obj_for_ilp_instance = obj_current_segment_for_log;
                    else:
                        print(f"Instance {entrada}: Segment ended with status: {current_segment_status_name}. No new feasible solution. Using last Obj: {obj_current_segment_for_log}");
                else:
                    print(f"Instance {entrada}: Solver returned no solution object (sol_segment is None). Status: {current_segment_status_name}. Using last Obj: {obj_current_segment_for_log}");

            except Exception as e_solve:
                print(f"Instance {entrada}: EXCEPTION during mdl.solve() or processing for segment {i_seg+1}: {e_solve}");
                current_segment_status_name = f"ExceptionInSolve: {type(e_solve).__name__}";

            cumulative_solver_time_for_ilp_instance += segment_actual_solve_time;

            gap_vs_bks_c = _calculate_gap_vs_bks_ilp(obj_current_segment_for_log, bks);
            if obj_current_segment_for_log is not None and gap_vs_bks_c is not None:
                current_obj_is_better_for_gap = False;
                if global_best_gap_vs_bks_for_ilp_instance is None:
                    current_obj_is_better_for_gap = True;
                elif gap_vs_bks_c < 0:
                    if global_best_gap_vs_bks_for_ilp_instance >= 0:
                        current_obj_is_better_for_gap = True;
                    elif gap_vs_bks_c > global_best_gap_vs_bks_for_ilp_instance:
                        current_obj_is_better_for_gap = True;
                elif gap_vs_bks_c >= 0 and gap_vs_bks_c < global_best_gap_vs_bks_for_ilp_instance:
                    current_obj_is_better_for_gap = True;

                if current_obj_is_better_for_gap:
                    global_best_gap_vs_bks_for_ilp_instance = gap_vs_bks_c;

            target_cumulative_time_log = chk_times[i_seg] if i_seg < len(chk_times) else cumulative_solver_time_for_ilp_instance;
            row = {
                'instance': entrada,
                'time': target_cumulative_time_log,
                'objective': obj_current_segment_for_log,
                'gap': gap_vs_bks_c,
                'best_gap': global_best_gap_vs_bks_for_ilp_instance,
                'best_obj': global_best_obj_for_ilp_instance,
            };
            log_data_for_this_ilp_instance.append(row);
            print(f"Instance {entrada}: Logged at time {target_cumulative_time_log:.2f}s. Obj: {obj_current_segment_for_log}, Gap: {gap_vs_bks_c}, BestGap: {global_best_gap_vs_bks_for_ilp_instance}, BestObj: {global_best_obj_for_ilp_instance}");

            if cumulative_solver_time_for_ilp_instance >= actual_time_limit - 1e-6 :
                print(f"Instance {entrada}: Cumulative solver time ({cumulative_solver_time_for_ilp_instance:.2f}s) reached total time limit ({actual_time_limit}s). Stopping iterative solve.");
                break;

    if log_data_for_this_ilp_instance:
        df_instance_checkpoints = pd.DataFrame(log_data_for_this_ilp_instance);
        write_header = not global_ilp_checkpoint_csv_path.exists() or global_ilp_checkpoint_csv_path.stat().st_size == 0;
        df_instance_checkpoints.to_csv(global_ilp_checkpoint_csv_path, mode='a', header=write_header, index=False);
        print(f"ILP Checkpoint data for instance {entrada} appended to {global_ilp_checkpoint_csv_path}");

    print(f"Total CPLEX solver time across all segments for ILP instance {entrada}: {cumulative_solver_time_for_ilp_instance:.2f} seconds.");

    if final_sol_object_from_iterative_solve:
        solucion = final_sol_object_from_iterative_solve;
        print(f"Instance {entrada}: Using final solution object from iterative solve. Associated status: {final_status_name_for_solucion}, Obj: {solucion.objective_value if hasattr(solucion, 'objective_value') and solucion.has_objective() else 'N/A'}");
    else:
        print(f"Instance {entrada}: No feasible solution found throughout iterative solving. 'solucion' will be an empty solution object.");
        solucion = mdl.new_solution();
        if not seg_durs:
            final_status_name_for_solucion = "NoSolveAttempted";
        else:
            final_status_name_for_solucion = mdl.get_solve_status().name if mdl.get_solve_status() else "StatusUnavailableAfterLoop";
    
    memoria_usada = np.around(psutil.Process().memory_info().rss / 10**6, decimals=2);
    print("Memoria usada:", memoria_usada, "megabytes");
    
    dict_sol["Memoria"] = [memoria_usada];

    if str(mdl.get_solve_status()) == "JobSolveStatus.OPTIMAL_SOLUTION":
        #list_sol[5] = "Opt";
        dict_sol["Status"] = ["Opt"];
    else:
        #list_sol[5] = "Fact";
        dict_sol["Status"] = ["Fact"];
        
    if mdl.get_solve_status().name == 'INFEASIBLE_SOLUTION':
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
        dfSolucion.to_excel(f"../output/LPM/ejec_output.xlsx", index=False);
        continue;

    try:
        solucion.display();
    except:
        dfAux = pd.DataFrame(dict_sol);
        dfSolucion = pd.concat([dfSolucion, dfAux]);
        dfSolucion.to_excel(f"../output/LPM/ejec_output.xlsx", index=False);
        print("No hay solución.");
        continue;

    print('1:',solucion.get_objective_value())
    #list_sol[1] = str(solucion.get_objective_value());
    dict_sol["Valor FO"] = [solucion.get_objective_value()];
    
    print('Best bound:',solucion.solve_details.best_bound)
    #list_sol[2] = str(round(solucion.solve_details.best_bound,2));
    dict_sol["Best Bound"] = [round(solucion.solve_details.best_bound, 2)];
    
    print('Rgap:',solucion.solve_details.mip_relative_gap)
    #list_sol[6] = str(round(solucion.solve_details.mip_relative_gap*100,2));
    dict_sol["Rel GAP"] = [round(solucion.solve_details.mip_relative_gap, 2)];
    
    #print('Time:',solucion.solve_details.time)

    time_fin = time.time();
    time_ejecucion = time_fin - time_inicio;
    #print("Tiempo ejecucion:",time_ejecucion)
    
    #list_sol[3] = str(tiempo_carga);
    dict_sol["Tiempo Carga"] = [tiempo_carga];
    
    #list_sol[4] = str(np.around(time_ejecucion, decimals=2));
    dict_sol["Tiempo Ejec"] = [round(time_ejecucion, 2)];

    pacientes_atend = [];
    fichas_ocup = [0 for i in range(len(surgeon))];
    cirujanos_atend_num = [0 for i in range(len(surgeon))];
    prioridad = 0;
    ocupacion = 0;
    ratios = [0 for i in range(len(surgeon))];
    PROG = {e:[None, None, None, None, None, None, None, None, None, None] for e in range(nSlot*nDays*len(room) + nDays*(len(room) + 1) + 2)};
    
    c = 0;
    for d in day: 
        PROG[c][0] = 'DIA ' + str(d);
        c += 1;
        for o in room: 
            PROG[c][0] = 'PABELLON ' + str(o);
            c += 1;
            for t in slot:
                for p in patient:
                    for s in surgeon:
                        #if version == "C":
                        #    fichas_ocup[s] = nFichas*nDays - f[(s,nDays-1)].solution_value;
                        for a in second:
                            if x[(p,o,s,a,t,d)].solution_value == 1:
                                PROG[c][0] = 'BLOQUE ' + str(t) 
                                PROG[c][1] = dfPatient.iloc[p]["nombre"];
                                if dfPatient.iloc[p]["id"] not in pacientes_atend:
                                    pacientes_atend.append(dfPatient.iloc[p]["id"]);
                                    fichas_ocup[s] += dictCosts[s, a, compress(o, d, t)];
                                    cirujanos_atend_num[s] += 1;
                                    prioridad += I[(p,d)];
                                    ocupacion += int(OT[p]);
                                PROG[c][2] = dfSurgeon.iloc[s][0];
                                PROG[c][3] = dfSecond.iloc[a][0];
                                PROG[c][4] = dfPatient.iloc[p]["especialidad"];
                                #PROG[c][5] = f[(s,d)].solution_value;
                                PROG[c][5] = dfdisAffi.iloc[a][s+1];
                                aux = 0;
                                for i in range(num_ext):
                                    aux += int(Ex[i][(s,WhichExtra(o,t//T,d,i)-1)]);
                                PROG[c][6] = aux;
                                PROG[c][7] = dfdisAffiDiario.iloc[d%5][s+1];
                                PROG[c][8] = dfdisAffiBloque.iloc[t//T][s+1];
                                PROG[c][9] = int(PROG[c][5]) + int(PROG[c][6]) + int(PROG[c][7]) + int(PROG[c][8]);
                                c += 1;         
    def export_results():
        pac = [];
        cir = [];
        sec = [];
        hab = [];
        dia = [];
        blo = [];
        for p in patient:
            for o in room:
                for s in surgeon:
                    for a in second:
                        for t in slot:
                            for d in day:
                                if x[(p,o,s,a,t,d)].solution_value == 1 and p not in pac:
                                    pac.append(p);
                                    cir.append(s);
                                    sec.append(a);
                                    hab.append(o);
                                    dia.append(d);
                                    blo.append(t);
        return pac, cir, sec, hab, dia, blo;
    
    print(export_results());
    
    dfPROG = pd.DataFrame({'BLOQUE': [PROG[c][0] for c in PROG],
        'PACIENTE': [PROG[c][1] for c in PROG],
        '1ER CIRUJANO': [PROG[c][2] for c in PROG],
        '2DO CIRUJANO': [PROG[c][3] for c in PROG],
        'TIPO PROC': [PROG[c][4] for c in PROG],
        'FICHAS CIR': [PROG[c][5] for c in PROG],
        'FICHAS EXT': [PROG[c][6] for c in PROG],
        'FICHAS DIA': [PROG[c][7] for c in PROG],
        'FICHAS BLO': [PROG[c][8] for c in PROG],
        'TOTAL FICHAS': [PROG[c][9] for c in PROG]});
    
    #dfSTATS = pd.DataFrame([[dfSurgeon.iloc[i][0] + ": ", str(f[(i,nDays-1)].solution_value)] for i in surgeon]);
    #print(dfSTATS);

    #writer = ExcelWriter(carpeta + "Resultados/Linear Programming Model/LPM C/" + "LPM_C_"+instancia[INS][0]+
    #                     "_"+instancia[INS][1]+"_"+"x"+"_"+instancia[INS][3]+"_"+instancia[INS][4]+".xlsx");
    
    dfPROG.to_excel(f"../output/LPM/{entrada}_Schedule_v{version}p{patient_code}n{nPatients}s{nSurgeons}d{nDays}.xlsx", index=False);
    #dfSTATS.to_excel(f"./Output/Stats_v{version}p{patient_code}s{nSurgeons}d{nDays}.xlsx", index=False);
    
    print("Pacientes atendidos:", len(pacientes_atend));
    
    #list_sol[7] = str(len(pacientes_atend));
    dict_sol["Pacientes Atend"] = [len(pacientes_atend)];

    dict_sol["Prioridad"] = [prioridad];
    
    #list_sol[8] = str(round(np.mean(fichas_ocup),1));
    print(fichas_ocup)
    dict_sol["Avg Fichas"] = [round(np.mean([i for i in fichas_ocup if i != 0]), 2)];
    
    #list_sol[9] = str(round(np.std(fichas_ocup),1));
    dict_sol["Std Fichas"] = [round(np.std([i for i in fichas_ocup if i != 0]), 2)];
    
    print(cirujanos_atend_num)
    #list_sol[10] = str(round(np.mean(cirujanos_atend_num),1));
    dict_sol["Avg Cirug"] = [round(np.mean([i for i in cirujanos_atend_num if i != 0]), 2)];
    
    #list_sol[11] = str(round(np.std(cirujanos_atend_num),1));
    dict_sol["Std Cirug"] = [round(np.std([i for i in cirujanos_atend_num if i != 0]), 2)];
    
    for i in range(nSurgeons):
        if cirujanos_atend_num[i] != 0:
            ratios[i] = round(fichas_ocup[i]/cirujanos_atend_num[i], 2);
            
    #list_sol[12] = str(round(np.mean(ratios),1));
    dict_sol["Avg Ratio"] = [round(np.mean([i for i in ratios if i != 0]), 2)];
    
    #list_sol[13] = str(round(np.std(ratios),1)) + "\n";
    dict_sol["Std Ratio"] = [round(np.std([i for i in ratios if i != 0]), 2)];

    dict_sol["Ocupación"] = [round(ocupacion/(nSlot*nDays*len(room)), 2)];
    
    dfAux = pd.DataFrame(dict_sol);
    dfSolucion = pd.concat([dfSolucion, dfAux]);
            
    dfSolucion.to_excel(f"../output/LPM/ejec_output.xlsx", index=False);
print('Fin del programa.');