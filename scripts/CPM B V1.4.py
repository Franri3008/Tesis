#!/usr/bin/env python
# coding: utf-8

# In[2]:


Potencia = 1;
exe_loc = "C:/Program Files/IBM/ILOG/CPLEX_Studio221/cpoptimizer/bin/x64_win64/cpoptimizer.exe";

from docplex.cp.model import CpoModel

import random
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from timeit import default_timer

rango = (1,-1); #Desde, hasta / 1 = primero, -1 = último. 2 números iguales: solo esa instancia
Potencia = 1;
carpeta = "C:/Users/HP/Cosas/Universidad/Proyecto/";

with open(carpeta+"TXT/instancias.txt") as a:
    instancia = a.readlines();
    for i in range(len(instancia)):
        instancia[i] = instancia[i].replace("\n","");
        instancia[i] = instancia[i].split(" ");
      
if rango[1] == -1:
    rango = (rango[0] - 1, len(instancia));
else:
    rango = (rango[0] - 1, rango[1] + 1);
    
for INS in range(rango[0],rango[1]):
    time_inicio = default_timer();
    print(f"Datos: {instancia[INS][0]}; Días: {instancia[INS][1]};\nAfinidad: {instancia[INS][2]}; Cirujanos: {instancia[INS][3]};\nFichas: {instancia[INS][4]}.");
    list_sol = [instancia[INS][0]+"_"+instancia[INS][1],"","","","","","","","","","","","","","",""];
    dfPrueba=pd.read_excel(open(carpeta + "Datos/" + instancia[INS][0]+'.xls', 'rb'),
                            sheet_name='Hoja1', converters={'n°':int}, index_col=[0])

    dfSurgeon=pd.read_excel(open(carpeta + "Datos/" + 'SURGEON.xlsx', 'rb'),
                            sheet_name='surgeon', converters={'n°':int}, index_col=[0])
    dfSecond=pd.read_excel(open(carpeta + "Datos/" + 'SECOND.xlsx', 'rb'),
                            sheet_name='second', converters={'n°':int}, index_col=[0])
    dfRoom=pd.read_excel(open(carpeta + "Datos/" + 'ROOM.xlsx', 'rb'),
                            sheet_name='room', converters={'n°':int}, index_col=[0])
    #dfTime=pd.read_excel(open(carpeta + "Datos/" + 'PROCESS_TIME.xls', 'rb'),
    #                        sheet_name='Process Time', converters={'n°':int}, index_col=[0])
    dfType=pd.read_excel(open(carpeta + "Datos/" + 'PROCESS_TYPE.xls', 'rb'),
                            sheet_name='Process Type', converters={'n°':int}, index_col=[0])

    #Antestesistas
    extras = [];
    #DISTRIBUCION QUIROFANOS
    dfdisORA=pd.read_excel(open(carpeta + "Datos/" + 'DISTRIBUCION OR_EXT.xlsx', 'rb'),
                            sheet_name='A', converters={'n°':int}, index_col=[0])

    dfdisORA=dfdisORA.astype(str).values.tolist()

    extraA = [];
    for i in range(len(dfdisORA)):
        aux = dfdisORA[i].copy();
        extraA.append(aux);

    for i in range(len(dfdisORA[0])):
        aux = dfdisORA[0][i];
        aux = aux.split(";");
        dfdisORA[0][i] = aux[0];
        extraA[0][i] = aux[1:];
        aux = dfdisORA[1][i];
        aux = aux.split(";");
        dfdisORA[1][i] = aux[0];
        extraA[1][i] = aux[1:];
        num_ext = len(extraA[1][i]);

    extras.append(extraA);

    dfdisORB=pd.read_excel(open(carpeta + "Datos/" + 'DISTRIBUCION OR_EXT.xlsx', 'rb'),
                            sheet_name='B', converters={'n°':int}, index_col=[0])
    dfdisORB=dfdisORB.astype(str).values.tolist()

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

    dfdisORC=pd.read_excel(open(carpeta + "Datos/" + 'DISTRIBUCION OR_EXT.xlsx', 'rb'),
                            sheet_name='C', converters={'n°':int}, index_col=[0])
    dfdisORC=dfdisORC.astype(str).values.tolist()

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

    dfdisORD=pd.read_excel(open(carpeta + "Datos/" + 'DISTRIBUCION OR_EXT.xlsx', 'rb'),
                            sheet_name='D', converters={'n°':int}, index_col=[0])
    dfdisORD=dfdisORD.astype(str).values.tolist()

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

    #dfDATA=pd.read_excel(open(carpeta + "Datos/" + 'IQX2018ELECTIVAS.xls', 'rb'),
    #                        sheet_name='ELECTIVAS', converters={'n°':int}, index_col=[0])

    dfdisAffi=pd.read_excel(open(carpeta + "Datos/" + 'AFFINITY_EXT.xlsx', 'rb'),
                            sheet_name='Hoja1', converters={'n°':int}, index_col=[0])
    
    dfdisAffiDiario=pd.read_excel(open(carpeta + "Datos/" + 'AFFINITY_DIARIO.xlsx', 'rb'),
                            sheet_name='Dias', converters={'n°':int}, index_col=[0])
    
    dfdisAffiBloque=pd.read_excel(open(carpeta + "Datos/" + 'AFFINITY_DIARIO.xlsx', 'rb'),
                            sheet_name='Bloques', converters={'n°':int}, index_col=[0])

    dfdisRank = pd.read_excel(open(carpeta + "Datos/" + 'RANKING.xlsx', 'rb'),
                             sheet_name = 'Hoja1', converters={'n°':int}, index_col=[0]);

    extra = [];
    dfExtra = [];
    for i in range(num_ext):
        aux = pd.read_excel(open(carpeta + "Datos/" + 'AFFINITY_EXT.xlsx', 'rb'),
                           sheet_name='Extra'+str(i+1), converters = {'n°':int}, index_col=[0]);
        extra.append(len(aux));
        dfExtra.append(aux);

    #Rankings para extras
    dfRankExtra = [];
    for i in range(num_ext):
        aux = pd.read_excel(open(carpeta + "Datos/" + 'RANKING.xlsx', 'rb'),
                           sheet_name='Extra'+str(i+1), converters = {'n°':int}, index_col=[0]);
        extra.append(len(aux));
        dfRankExtra.append(aux);

    #dfType=dfType.iloc[:5]
    #dfRoom=dfRoom.iloc[:1]
    dfPatient=dfPrueba
    #dfPrueba=dfPrueba.astype(str).values.tolist()
    #numPat=100
    #dfPatient=dfPatient.iloc[:numPat]

    #%%
    #Indices
    random.seed(0)
    patient=[p for p in range(len(dfPatient))]
    nSurgeon = int(instancia[INS][3]);
    surgeon=[s for s in range(nSurgeon)];
    #second=[a for a in range(len(dfSecond))]
    second = [a for a in range(nSurgeon)]
    room=[o for o in range(len(dfRoom))]
    nDay = int(instancia[INS][1]);
    day=[d for d in range(nDay)];
    nSlot=8  #bloques de 60 minutos
    slot=[t for t in range(nSlot)]
    nLEN=8
    ES=nSlot-nLEN
    process=[t for t in range(len(dfType))]

    #level_affinity = 0  #1 baja afinidad - 7 alta afinidad
    level_affinity = int(instancia[INS][2]);

    #Arcos #probar np
    arcox=[(p,o,s,a,t,d) for p in patient for o in room for s in surgeon for a in second for t in slot for d in day]
    arcoy=[(p,o,d) for p in patient for o in room for d in day]
    arcoz=[(p,s,a) for p in patient for s in surgeon for a in second]
    arcot=[(p) for p in patient];
    arcof=[(s,d) for s in surgeon for d in [a for a in range(-1,nDay)]];

    #dfdisAffi.iloc[s][a+1]

    #%%
    #Max surgeries and budget per surgeon
    M=np.zeros(len(surgeon), dtype=int)
    Pr = np.zeros(len(surgeon), dtype=int);
    for s in surgeon:
        M[s]=int(dfSurgeon.iloc[s][8])
        Pr[s] = int(dfSurgeon.iloc[s][11]);

    E=np.ones((len(slot),len(day)))*1000
    A=np.ones((len(slot),len(day)))*1000
    B=np.ones(len(patient))*1
    Y=np.ones(len(patient))*1

    #%%
    #Parametros
    #Create priorities
    I=np.ones((len(patient),len(day)), dtype=int)
    for p in range(len(dfPatient)):
        for d in day:
            try:
                I[(p,d)]=1 + dfPatient.iloc[p][0]*dfPatient.iloc[p][4]*0.0001/(d+1)
            except ValueError:
                print("Value Error en calculo de prioridades.")

    #Overtime cost
    #OTC=np.ones(len(room))*0

    #Matriz de coincidencia cirujanos
    COIN = np.zeros((len(surgeon),len(second)),dtype=int)
    for s in surgeon:
        for a in second:
            if dfSurgeon.iloc[s][0]==dfSecond.iloc[a][0]:
                COIN[(s,a)]=1
            else :
                COIN[(s,a)]=0


    A=np.ones((len(surgeon),len(second)), dtype=int)
    for s in surgeon:
        for a in second:
            A[(s,a)]=dfdisAffi.iloc[s][a+1];

    Ex = [np.ones((len(surgeon),extra[i]),dtype=float) for i in range(num_ext)];
    for i in range(num_ext):
        for s in surgeon:
            for e in range(extra[i]):
                Ex[i][(s,e)] = dfExtra[i].iloc[e][s+1];

    #Surgeons availability
    '''
    SD = np.zeros((len(surgeon),len(second),len(slot),len(day)), dtype=int)
    for s in surgeon:
        for a in second:
            for d in day:
                if dfSurgeon.iloc[s][2+d]==1:
                    if dfSecond.iloc[a][2+d]==1:
                        if dfSurgeon.iloc[s][1]==1:
                            if dfSecond.iloc[a][1]==1:
                                for t in range(0,int(nSlot/2)):
                                    SD[(s,a,t,d)]=1
                        if dfSurgeon.iloc[s][2]==1:
                            if dfSecond.iloc[a][2]==1:
                                for t in range(int(nSlot/2),nSlot):
                                    SD[(s,a,t,d)]=1
    '''
                                    
    SDm = np.zeros((len(surgeon),len(slot),len(day)), dtype=int);
    for s in surgeon:
        for d in day:
            if dfSurgeon.iloc[s][2+d] == 1:
                if dfSurgeon.iloc[s][1] == 1:
                    for t in range(0,int(nSlot/2)):
                        SDm[(s,t,d)] = 1;
                if dfSurgeon.iloc[s][2] == 1:
                    for t in range(int(nSlot/2),nSlot):
                        SDm[(s,t,d)] = 1;
                        
    SDs = np.zeros((len(second),len(slot),len(day)), dtype=int);
    for a in second:
        for d in day:
            if dfSecond.iloc[a][2+d] == 1:
                if dfSecond.iloc[a][1] == 1:
                    for t in range(0,int(nSlot/2)):
                        SDs[(a,t,d)] = 1;
                if dfSecond.iloc[a][2] == 1:
                    for t in range(int(nSlot/2),nSlot):
                        SDs[(a,t,d)] = 1;   

    #Availability of patient
    DISP=np.ones(len(patient))
    #%%
    def busquedaType(especialidad):
        indice=0
        for i in range(len(process)):
            if (especialidad == dfType.iloc[i][0]):
                indice = i
                #print(indice)
        return indice
    dfType.iloc[1][0]
    #Patient and surgeon compatibility
    SP=np.zeros((len(patient),len(surgeon)),dtype=int)
    contador=0
    for p in patient:
        #print(dfPatient.iloc[p][21])
        for s in surgeon:
            if busquedaType(dfPatient.iloc[p][21]) == busquedaType(dfSurgeon.iloc[s][9]):
                #print(dfPatient.iloc[p][21])
                SP[p][s]=1
    #%%               

    #Crea diccionario de paciente 
    dic_p={e:[] for e in patient}
    for p in patient:
        dic_p[p] = [0,0,0,0,0]
        #dic_p[p][0] = list_patient[p] #paciente y número aleatorio asociado (entre 0 y 1)
        dic_p[p][1] = busquedaType(dfPatient.iloc[p][21]) #id de tipo de especialidad
        dic_p[p][2] = dfPatient.iloc[p][2] #nombre paciente
        dic_p[p][3] = p #indice del paciente
        dic_p[p][4] = dfPatient.iloc[p][21] #nombre especialidad

    #Room and patient compatibility
    AOR=np.zeros((len(patient),len(room),len(slot),len(day)))
    dicOR={e:[] for e in range(len(dfRoom))}
    j=[]
    z=[]
    ns=0
    for o in room:
        if o==0:
            for d in day:
                for e in range(2):
                    #print(e)
                    if e==0:                    
                        for t in range(0,int(len(slot)/2)):

                            j=dfdisORA[e][d]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
                    if e==1:
                        #print('paso')
                        for t in range(int(len(slot)/2),len(slot)):

                            j=dfdisORA[e][d]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
        if o==1:
            for d in day:
                for e in range(2):
                    if e==0:                    
                        for t in range(0,int(len(slot)/2)):
                            j=dfdisORB[e][d]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
                    if e==1:
                        for t in range(int(len(slot)/2),len(slot)):
                            j=dfdisORB[e][d]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
        if o==2:
            for d in day:
                for e in range(2):
                    if e==0:                    
                        for t in range(0,int(len(slot)/2)):
                            j=dfdisORC[e][d]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
                    if e==1:
                        for t in range(int(len(slot)/2),len(slot)):
                            j=dfdisORC[e][d]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
        if o==3:
            for d in day:
                for e in range(2):
                    if e==0:                    
                        for t in range(0,int(len(slot)/2)):
                            j=dfdisORD[e][d]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
                    if e==1:
                        for t in range(int(len(slot)/2),len(slot)):
                            j=dfdisORD[e][d]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1

    p=0
    o=0
    t=0

    for ns in range(len(dic_p)):
        for nP in range(len(dicOR)):
            #print(dicOR[nP][3])
            if str(dic_p[ns][1])==dicOR[nP][3]:
                p=dic_p[ns][3]
                o=dicOR[nP][0]
                t=dicOR[nP][2]
                d=dicOR[nP][1]
                AOR[p][o][t][d]=1;   

    OT=np.zeros(len(patient))
    for p in patient:
        OT[p] = dfPatient.iloc[p][22];

    #%%
    print('Datos Obtenidos')

    #########################################################################################################################
    #########                                      Contruccion modelo cplex                                         #########
    #########################################################################################################################
    inicio_carga = default_timer();
    
    from docplex.cp.model import *
    import docplex.cp.solution as Solucion

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
    
    T = [integer_var(name="T_"+str(p),min=0,max=nDay*len(slot)-1) for p in patient];
    
    B = nDay * len(slot);
    
    print("Cargando función objetivo...")
    
    #mdl.add(maximize(sum(I[(p,T[p]//nDay)] for p in patient if T[p] != -1)));   
    mdl.add(maximize(sum(1000 * I[(p,d)] * C[p][d] for p in patient for d in day)));
    
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
            
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
            
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
                    if AOR[(p,o,t,d)] == 0:
                        mdl.add(mdl.if_then(mdl.logical_and(O[p] == o, T[p] == d*len(slot) + t),
                                           O[p] == len(room)));
                
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
    
    #9: Disponibilidad cirujanos y paciente
    print("Cargando 9...");
    for p in patient:
        for s in surgeon:
            for t in slot:
                for d in day:
                    if SDm[(s,t,d)] == 0:
                        mdl.add(mdl.if_then(T[p] == d*len(slot) + t,
                                           M[p] != s));
                        
    for p in patient:
        for a in second:
            for t in slot:
                for d in day:
                    if SDs[(a,t,d)] == 0:
                        mdl.add(mdl.if_then(T[p] == d*len(slot) + t,
                                           S[p] != a));                   
                            
    for p in patient:
        if DISP[p] == 0:
            mdl.add(M[p] == len(surgeon));
            
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
    
    #10: No puede usarse un bloque de la mañana y de la tarde para una operación
    for p in patient:
        mdl.add((T[p] % len(slot))//4 == ((T[p] + int(OT[p]) - 1) % len(slot))//4);
        
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
    
    #11: Afinidad
    print("Cargando 10...");
    
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
                        e = int(extras[o][t][d][i]);
                        if Ex[i][(s,e-1)] < level_affinity:
                            for p in patient:
                                mdl.add(mdl.if_then(M[p] == s, mdl.logical_or(O[p] != o, 
                                                    mdl.logical_or(T[p] < d*len(slot) + t * 4, 
                                                    T[p] >= d*len(slot) + (t + 1) * 4))))
    
    for s in surgeon:
        for d in day:
            if dfdisAffiDiario.iloc[d][s+1] < level_affinity:
                for p in patient:
                    mdl.add(mdl.if_then(M[p] == s, mdl.logical_or(T[p] < d*len(slot), T[p] >= (d+1)*len(slot))));
                    
    for s in surgeon:
        if dfdisAffiBloque.iloc[0][s+1] < level_affinity:
            for p in patient:
                mdl.add(mdl.if_then(M[p] == s, T[p] % len(slot) > 3));
        if dfdisAffiBloque.iloc[1][s+1] < level_affinity:
            for p in patient:
                mdl.add(mdl.if_then(M[p] == s, T[p] % len(slot) <= 3));
                    
    print("Restricciones:",CpoModelStatistics(mdl).get_number_of_constraints());
    
    fin_carga = default_timer();
    tiempo_carga = str(np.around(fin_carga - inicio_carga, decimals=2));
    #Resolución
    print('Restricciones Cargadas.')
    print('Resolviendo...')
    time_inicio = default_timer();
    
    if Potencia == 1:
        solucion = mdl.solve(TimeLimit=1800,Workers=1);
    elif Potencia == 2:
        solucion = mdl.solve(TimeLimit=3600);
                            
    infos = solucion.get_solver_infos();
    info_memoria = (infos['MemoryUsage'],infos['PeakMemoryUsage']);  # Display memory usage of the CP engine
    list_sol[14] = str(np.around(info_memoria[0]/10**6, decimals=2));
    list_sol[15] = str(np.around(info_memoria[1]/10**6, decimals=2)) + "\n";
    print("info_memoria:",info_memoria,"bytes.");
    print("-----")

    #print('status:',mdl.get_solve_status());
    if solucion.is_solution():
        if solucion.is_solution_optimal():
            list_sol[5] = "Opt";
        else:
            list_sol[5] = "Fact";
    else:
        list_sol[5] = "NS";
        list_sol[1] = "-";
        list_sol[2] = "-";
        list_sol[6] = "-";
        time_fin = default_timer()
        time_ejecucion = time_fin - time_inicio
        print("Tiempo ejecucion:",time_ejecucion);
        list_sol[3] = str(tiempo_carga);
        list_sol[4] = str(np.around(time_ejecucion, decimals=2));
        list_sol[7] = "0";
        list_sol[10] = "0";
        list_sol[11] = "0";

        from os.path import exists;
        existe_txt = exists(carpeta + "TXT/CPM/" + "CPM_B" + ".txt");
        print(existe_txt);
        if existe_txt == False:
            aux = open(carpeta + "TXT/CPM/" + "CPM_B" + ".txt","w");
            aux.close();

        def escribir(nuevo,txt):
            for i in range(len(txt)):
                if nuevo[0] == txt[i][0]:
                    return txt;
            txt.append(nuevo);
            return txt;
        with open(carpeta + "TXT/CPM/" + "CPM_B" + ".txt","r") as a:
            texto = a.readlines();
            for i in range(len(texto)):
                texto[i] = texto[i].split(" ");
        with open(carpeta + "TXT/CPM/" + "CPM_B" + ".txt","w") as a:
            a.write("");
        with open(carpeta + "TXT/CPM/" + "CPM_B" + ".txt","a") as a:
            texto = escribir(list_sol,texto);
            for i in texto:
                i = " ".join(i);
                a.write(i)
        continue;

    #print("Número de soluciones:",solucion.CpoSolverInfos.get_number_of_solutions());

    #print('1:',solucion.get_objective_value())
    list_sol[1] = str(solucion.get_objective_value());
    #print('Best bound:',solucion.solve_details.best_bound)
    list_sol[2] = str(solucion.get_objective_bound());
    #print('Rgap:',solucion.solve_details.mip_relative_gap)
    list_sol[6] = str(np.around(solucion.get_objective_gap(),decimals=2));
    #print('Time:',solucion.solve_details.time)

    time_fin = default_timer()
    time_ejecucion = time_fin - time_inicio
    print("Tiempo ejecucion:",time_ejecucion);
    
    if not solucion:
        mdl.refine_conflict().write()
        stop
    
    list_sol[3] = str(tiempo_carga);
    list_sol[4] = str(np.around(time_ejecucion, decimals=2));
    #%%
    pacientes_atend = [];
    fichas_ocup = [0 for i in range(len(surgeon))];
    cirujanos_atend_num = [0 for i in range(len(surgeon))];
    ratios = [0 for i in range(len(surgeon))];
    #PROG={e:[None, None, None, None, None] for e in range(100)}
    PROG = {e:[None, None, None, None, None, None, None, None, None, None] for e in range(nSlot * nDay * len(room) + nDay*(len(room) + 1) + 20)};
    c=0
    lista_val = [];
    lista_p = [];
    for p in patient:
        val = solucion.get_solution().get_value(T[p]);
        if val >= 0:
            lista_val.append(val);
            print(f"Sol var O[{p}] = {solucion.get_solution().get_value(O[p])}");
            print(f"Sol var T[{p}] = {solucion.get_solution().get_value(T[p])}");
            print(f"Sol var M[{p}] = {solucion.get_solution().get_value(M[p])}");
            print(f"Sol var S[{p}] = {solucion.get_solution().get_value(S[p])}");
            lista_p.append(p);
        else:
            print(f"Sol var O[{p}] = {solucion.get_solution().get_value(O[p])}");
            print(f"Sol var T[{p}] = {solucion.get_solution().get_value(T[p])}");
            print(f"Sol var M[{p}] = {solucion.get_solution().get_value(M[p])}");
            print(f"Sol var S[{p}] = {solucion.get_solution().get_value(S[p])}");
    lista_p = [x for _,x in sorted(zip(lista_val,lista_p))];
    
    print("Lista p:", lista_p);
    
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
                        if dfPatient.iloc[p][2] not in pacientes_atend:
                            pacientes_atend.append(dfPatient.iloc[p][2]);
                            fichas_ocup[s] += dfdisAffi.iloc[sec][surg+1] + dfdisAffiDiario.iloc[d][surg+1] + dfdisAffiBloque.iloc[tiempo % len(slot)//4][surg+1];
                            for i in range(num_ext):
                                fichas_ocup[s] += int(Ex[i][(surg,int(WhichExtra(o,tiempo % len(slot)//4,d,i)-1))]);        
                            cirujanos_atend_num[surg] += 1;
                        PROG[c][2]= dfSurgeon.iloc[surg][0];
                        PROG[c][3]= dfSecond.iloc[sec][0];
                        PROG[c][4]= dfPatient.iloc[p][21];
                        PROG[c][5] = dfdisAffi.iloc[surg][sec+1];
                        aux = 0;
                        for i in range(num_ext):
                            aux += int(Ex[i][(surg,WhichExtra(o,(tiempo % len(slot))//4,d,i)-1)]);
                        PROG[c][6] = aux;
                        PROG[c][7] = dfdisAffiDiario.iloc[d][surg+1];
                        PROG[c][8] = dfdisAffiBloque.iloc[(tiempo % len(slot))//4][surg+1];
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
    
    #dfSTATS = pd.DataFrame([[dfSurgeon.iloc[i][0] + ": ",str(solucion.get_solution().get_value(f[i][nDay]))] for i in surgeon]);
    #print(dfSTATS);

    writer = ExcelWriter(carpeta + "Resultados/Constraint Programming/CPM B/" + "CPM_B_" +instancia[INS][0]+"_"+instancia[INS][1]+"_"+instancia[INS][2]+"_"+instancia[INS][3]+".xlsx");
    dfPROG.to_excel(writer, 'PROGRAMACION QX', index=False);
    #dfSTATS.to_excel(writer, 'STATS CIRUJANOS', index=False);
    writer.save();
    print("Pacientes atendidos:",len(pacientes_atend));
    list_sol[7] = str(len(pacientes_atend));
    list_sol[8] = str(np.mean(fichas_ocup));
    list_sol[9] = str(np.std(fichas_ocup));
    list_sol[10] = str(np.mean(cirujanos_atend_num));
    list_sol[11] = str(np.around(np.std(cirujanos_atend_num), decimals=2));
    for i in range(len(surgeon)):
        if cirujanos_atend_num[i] != 0:
            ratios[i] = np.around(fichas_ocup[i]/cirujanos_atend_num[i], decimals=2);
    list_sol[12] = str(np.mean(ratios));
    list_sol[13] = str(np.std(ratios));
    
    from os.path import exists;
    existe_txt = exists(carpeta + "TXT/CPM/" + "CPM_B_" + str(level_affinity) + ".txt");
    print(existe_txt);
    if existe_txt == False:
        aux = open(carpeta + "TXT/CPM/" + "CPM_B_" + str(level_affinity) + ".txt","w");
        aux.close();
        
    def escribir(nuevo,txt):
        for i in range(len(txt)):
            if nuevo[0] == txt[i][0]:
                return txt;
        txt.append(nuevo);
        return txt;
    with open(carpeta + "TXT/CPM/" + "CPM_B_" + str(level_affinity) + ".txt","r") as a:
        texto = a.readlines();
        for i in range(len(texto)):
            texto[i] = texto[i].split(" ");
    with open(carpeta + "TXT/CPM/" + "CPM_B_" + str(level_affinity) + ".txt","w") as a:
        a.write("");
    with open(carpeta + "TXT/CPM/" + "CPM_B_" + str(level_affinity) + ".txt","a") as a:
        texto = escribir(list_sol,texto);
        for i in texto:
            i = " ".join(i);
            a.write(i)
print('Fin del programa;');


# In[ ]:




