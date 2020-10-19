# -*- coding: utf-8 -*-
#!/usr/bin/env python

######################### ...
# BasicCaudCR2_v200
#	13.10.2020

## Descripción: 
# 	Librería que contiene fuciones que realizan tareas básicas con caudales instantáneos del CR2: 
#		importación, revisión de calidad, gráfica y correlaciones.


## Requerimientos:
# 
#		Versión python:												# Python 3.7.3rc	

##
# Librerías

import os														# Python 3.7.3rc	
from os import walk													# Python 3.7.3rc
import os.path															# Python 3.7.3rc 
import sys																	# Python 3.7.3rc	
import time																	# Python 3.7.3rc
import itertools														# Python 3.7.3rc  

import numpy as np													# numpy 1.16.2

import matplotlib.pyplot as plt 						# matplotlib 3.0.2	for i in range(len(f)):
	
import matplotlib as mpl										# matplotlib 3.0.2
from matplotlib import cm										# matplotlib 3.0.2
import matplotlib.dates as mdates						# matplotlib 3.0.2

import xlrd																	# xlrd 1.2.0
import xlwt																	# xlwt 1.3.0
from xlutils import copy										# xlutils 2.0.0

import pandas as pd													# pandas 0.24.2
from pandas.plotting import scatter_matrix 	# pandas 0.24.2 

from sklearn import linear_model						# sklearn 0.21.2

import matplotlib.dates as mdates
#from matplotlib.ticker import MultipleLocator
#from matplotlib.ticker import IndexLocator 

import seaborn as sns



def creaCarpeta(ruta):
    if os.path.exists(ruta):
        print('La ruta existe, el algoritmo escribirá / quizá sobreescribirá en la carpeta:', '\n ', ruta)
    else:
        print('Para guardar los datos se crea la carpeta:', '\n ', ruta)
        os.mkdir(ruta)


########################## ...
## Función importa_Q_CR2 
#  versión 004 / 05.2020


def importa_Q_CR2(ruta_archivos, rutaGuardar, fechaIni, fechaFin ,listaEstaciones, txtDescr):
    
    
    directorio = ruta_archivos
    if directorio[-1] != '/':
    	directorio = directorio+'/'
    
    ruta = []
    f = []
    dn = []
    
    ruta_completa = []
    archivo = []
    for (dirpath, dirnames, filenames) in walk(directorio):
        dp = []
        f = []
        f.extend(filenames)
        dn.extend(dirnames)
        dp.extend(dirpath)
        directorio_aux = "".join(dp)
    
    archivo1 = 'cr2_qflxDaily_2018_stations.txt'
    archivo2 = 'cr2_qflxDaily_2018.txt'
    
    infoEsta = pd.read_csv(directorio+archivo1, index_col='codigo_estacion')
    #print(infoEsta)
    
    ###···encabez = pd.read_csv(directorio+archivo2,header=0,nrows=15)
    ###···print(encabez)
    #input('encabez')
    
    
    Qcrudos = pd.read_csv(directorio+archivo2,header=0,skiprows=range(1,15))
    Qcrudos.rename(columns={'codigo_estacion':'Fecha'}, inplace=True)
    indice = pd.to_datetime(Qcrudos['Fecha'])
    Qcrudos.index = indice
    Qcrudos.drop('Fecha', axis=1, inplace=True)
    
    #print(Qcrudos.columns.values)  
    
    listaEstacionesExistentes_int = []
    listaEstacionesExistentes_str = []
    
    for est in listaEstaciones:
        encontrada = False
        #print('buscando... ', est)
        for i in range(len(Qcrudos.columns.values)):
            if est in str(Qcrudos.columns.values[i]):
                print('La estación ', est,' está OK')
                encontrada = True
                listaEstacionesExistentes_int.append(int(est))
                listaEstacionesExistentes_str.append(est)
                
            elif (i == len(Qcrudos.columns.values)-1 and encontrada == False):
                print('La estación ', est,' no está  noooooooooooooooo', ' - nombre')
    
    fechaIni = pd.to_datetime(fechaIni, dayfirst=True)
    fechaFin = pd.to_datetime(fechaFin, dayfirst=True)
    
    #print(listaEstacionesExistentes_str[0])
    #print(type(listaEstacionesExistentes_str[0]))
    
    Qdia = Qcrudos[listaEstacionesExistentes_str][fechaIni:fechaFin]
    #Qdia = Qcrudos[listaEstacionesExistentes_str[0]][fechaIni:fechaFin]
    print(Qdia)
  
    indice_rellenado = pd.date_range(start=fechaIni,end=fechaFin,freq='D')
    #indice_rellenado = pd.date_range(start='01/01/1910',end='01/01/2020',freq='D')
    Qdia = Qdia.reindex(index=indice_rellenado)

    Qdia_gap = Qdia[Qdia < 0]


    for est in Qdia.columns.values:
        #· Se busca si existe algún dato válido
        if Qdia[est].first_valid_index() == None:
            print('La estación '+ str(est) + ' no tiene datos válidos en el período seleccionado')
            Qdia_gap[est].replace(-9999, -9, inplace=True)
            #Qdia_gap[est].replace(True, -9, inplace=True)
            Qdia[est].replace(-9999,np.nan, inplace=True)

        else:

            maxValor = Qdia[est].max()
            valorGap =  -maxValor*0.05
            Qdia_gap[est].replace(-9999, valorGap, inplace=True)
            #Qdia_gap.replace(True, -valorGap, inplace=True)
            Qdia[est].replace(-9999,np.nan, inplace=True)
    
    
    #print(infoEsta.loc[listaEstacionesExistentes_int])
    
    #print(Qdia)
    #print(Qdia_gap)
    #input()
    

    #calidad['Nombre']= infoEsta['nombre'].values
    params = {'legend.fontsize': 7,
             'axes.labelsize': '7',
             'axes.titlesize':'7',
             'xtick.labelsize':'7',
             'ytick.labelsize':'7'}

    mpl.rcParams.update(params)

    font = {'family' : 'normal',
            'weight' : 'regular',
            'size'   : 7}
    
    mpl.rc('font', **font)
    
    #mpl.rcParams.update({'font.size': 8})
    #n_grafs = 1
    n_grafs = len(listaEstacionesExistentes_str)
    #fig, axes = plt.subplots(n_grafs,1,sharex=True,sharey=True,dpi=300)
    
    #### Gráfica de series temporales marcando gaps ###
    fig = plt.figure(constrained_layout=True,figsize=(7.5,0.95*n_grafs)) 
    gs = fig.add_gridspec(n_grafs,1)
    for ii in range(n_grafs):
        ax = fig.add_subplot(gs[ii,0])
        #Qdia[listaEstacionesExistentes_str[ii]].plot(color='#0066FF', linestyle='-', linewidth=0.4,figsize=(12,3),legend=True, ax=ax)
        #Qdia[listaEstacionesExistentes_str[ii]].plot(color='#0066FF', linestyle='-', linewidth=0.4,legend=True, ax=ax)
        Qdia[listaEstacionesExistentes_str[ii]].plot(color='#0066FF', linestyle='-', linewidth=0.4, ax=ax)
        Qdia_gap[listaEstacionesExistentes_str[ii]].plot(linestyle = '-',linewidth=0.4, c='#EF4477' , ax = ax)
        #Qdia_gap[listaEstacionesExistentes_str[ii]].plot(style = '.', markersize = 0.1, c='#EF4477' , ax = ax)
        ax.grid(b=True, which='minor', color='0.75', linestyle='-',linewidth=0.2)
        ax.minorticks_on()
        ax.grid(b=True, which='major', color='0.60', linestyle='-', linewidth=0.4)
        #ax.set_ylabel('Caudal medio diario $[m^3/s]$')
        ax.set_title(listaEstacionesExistentes_str[ii],loc='left')
        #ax.set_title(nombres_estaciones[0],loc='left')
        if ii != n_grafs-1:
            ax.set_xticklabels([])
    
    plt.savefig(rutaGuardar + txtDescr+'_Qmd_SerieTemporalConGaps.png',dpi=300,bbox_inches='tight')
    
    
    
    print(Qdia)
    #input('qqdis')
    return Qdia



## Qdia es un data frame con una o varias columnas (estaciones) con fechas de índice
def calidadEstadistica(Qdia, rutaGuardar,txtDescr):

    calidad = pd.DataFrame(columns=['ID','Nombre','Año inicio','Año final','Extension [año]','Ndatos','Nvalidos','Nnulos','% Datos faltantes'],index=Qdia.columns.values)
    for est in Qdia.columns.values:


        if Qdia[est].first_valid_index() == None:

            print('La estación '+ str(est) + ' no tiene datos válidos en el período seleccionado')

        else:

    	    primerdia = Qdia[est].first_valid_index()
    	    ultimodia = Qdia[est].last_valid_index()
    	    anho_inicial = primerdia.year
    	    anho_final = ultimodia.year
    
    
    
    	    Ndatos = len(Qdia[est][primerdia:ultimodia])
    	    Nvalidos = len(Qdia[est][primerdia:ultimodia].loc[Qdia[est] >= 0])
    
    
    
    	    calidad['Ndatos'][est]=Ndatos
    	    calidad['Nvalidos'][est]=Nvalidos
    	    calidad['Nnulos'][est]=Ndatos - Nvalidos
    	    calidad['Año inicio'][est]=  anho_inicial
    	    calidad['Año final'][est]= anho_final
    	    calidad['Extension [año]'][est]= anho_final - anho_inicial
    	    #calidad['% Datos faltantes'][est[i]]= (Ndatos - Nvalidos)/Ndatos
    	    calidad['% Datos faltantes'][est]= 100 * (Ndatos - Nvalidos)/Ndatos
    	    #calidad['Nombre'][est[i]]= encabez[est[i]][2]
    	    
    	    print(calidad)
    #input('corta')
    #print(calidad.index.name)
    calidad.index.name = 'Código BNA'
    #print(calidad.to_latex())
    #tablaLatex(calidad[calidad.columns.values[[1,4,5,8]]],rutaGuardar,txtDescr)


    Qdia.to_excel(rutaGuardar + txtDescr+'_Qmd_EstInteres.xlsx')
    
    calidad.to_excel(rutaGuardar +txtDescr+'_CalidadCaudales.xlsx')
    #print(Qdia)
    #input('Qdia')
    
    extension = pd.DataFrame() 
    gaps = Qdia.isna()
    
    #print(gaps)
    #input('gaps')
    extension = gaps.replace(True, np.nan)
    #print(extension)
    #input('extension')
    nEstaciones = len(Qdia.columns.values)

    params = {'legend.fontsize': 8,
             'axes.labelsize': '8',
             'axes.titlesize':'8',
             'xtick.labelsize':'8',
             'ytick.labelsize':'8'}

    mpl.rcParams.update(params)

    font = {'family' : 'normal',
            'weight' : 'regular',
            'size'   : 8}
    
    mpl.rc('font', **font)
    #figura0, ax0 = plt.subplots(figsize=(10,0.5*nEstaciones))
    figura0, ax0 = plt.subplots(figsize=(8,nEstaciones * 0.5))
    #figura0, ax0 = plt.subplots(figsize=(10,2))
    
    for i in range(len(Qdia.columns.values)):
    	extension[Qdia.columns.values[i]].replace(0, i+1, inplace=True)
    	extension.plot(y=Qdia.columns.values[i], linewidth=20, ax=ax0, color='#1f64cc', legend=False)
    
    extension.to_excel(rutaGuardar +txtDescr+'_ExtensionCaud.xlsx')
    
    ax0.set_ylim((-0.02, len(Qdia.columns.values)+1.02))
    ax0.set_ylabel(txtDescr, fontsize=8)
    ax0.set_xlabel('Tiempo', fontsize=8, color=(0,0,0))
    
    ax0.grid(b=True, which='major', color='0.60', linestyle='-', linewidth=0.6)
    ax0.grid(b=True, which='minor', color='0.75', linestyle='-',linewidth=0.3)
    plt.yticks(range(1,len(Qdia.columns.values)+1),calidad.index.values)#, rotation=55)
    plt.title(txtDescr+' Diagrama de barras de extensión de estadística')
    plt.savefig(rutaGuardar +txtDescr+'_ExtensionEstadistCaudal.png', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()



def AnFrec_weibull(Qdia,rutaGuardar):  #####

    indOrig = Qdia.index
    #print(Qdia)
    #print(type(Qdia))
    #input('tipo')
    ####···#estSintet = str(Qdia.columns.values[0]) +'+'+str(Qdia.columns.values[1])
    #print(estSintet)
    #print(Qdia.isna())
    Qnulos = Qdia.isna()
    
    #print(Qnulos)
    
    #print(Qnulos.loc[Qnulos[Qnulos.columns.values[1]] == True])
    
    
    ####···#indiceNan_Areemplazar = Qnulos.loc[Qnulos[Qnulos.columns.values[1]] == True].index
    
    
    
    #print(Qnulos.loc[Qnulos[Qnulos.columns.values[0]] == False])
    #print(Qnulos.loc[Qnulos[Qnulos.columns.values[0]].values == False and Qnulos[Qnulos.columns.values[1]].values == True])
    #print(indiceNan_Areemplazar)
    
    
    #print(Qdia[Qdia.columns.values[1]])
    
    #input('Qdia 1 antes')
    #Qdia[Qdia.columns.values[1]][indiceNan_Areemplazar] == 0
    
    Qdia.dropna(inplace=True, how='all') ##
    #print(Qdia[Qdia.columns.values[1]])
    #input('Qdia 1 desp')
    #print(Qdia)
    
    #### Para hacer la suma de dos columnas (dos estaciones) 
    ####···##Qdia[estSintet] = Qdia.sum(axis=1) ## Por defecto este método ignora los valores nulos
    
    #print(Qdia)
    Qdia = Qdia.reindex(indOrig)
    #print(Qdia)
    #input('Qdia')
    
    ####···##estProba.append(estSintet)
    #### Para hacer la suma de dos columnas (dos estaciones) 
    
    #print(Qdia)
    #print(estProba)
    

    
    
    comentarios = open(rutaGuardar+'_comentarioSalidas' ,'w') #· crea archivo para comentarios o errores
    for est in Qdia.columns.values:
        txtDescr = 'Est'+str(est)
        qMedAn = []
        anhos = []
        
        ####····fechaIni = pd.to_datetime(fechaIni, dayfirst=True)
        ####····fechaFin = pd.to_datetime(fechaFin, dayfirst=True)
        ####····print(fechaIni)
        ####····print(fechaFin)
    
        if Qdia[est].first_valid_index() == None:
            comentario = 'La estación '+ str(est) + ' no tiene datos válidos en el período seleccionado'
            comentarios.write(comentario)


        else:
            fechaIni = Qdia[est].first_valid_index()
            fechaFin = Qdia[est].last_valid_index()
    
            ##print(fechaIni)
            #print(fechaFin)
            indDia_ts = pd.date_range(fechaIni,fechaFin,freq='d')
            
            anhoIni = int(fechaIni.year)
            anhoFin = int(fechaFin.year)
    
    
    
    
            for t in range(anhoIni, anhoFin+1):
                
                #qMedAn_t = Qdia[Qdia.columns.values[0]][str(t)].mean()
                qMedAn_t = Qdia[est][str(t)].mean()
                
                qMedAn.append(qMedAn_t)
    
    
                #if t == anhoIni:
                #    qMedAn = [qMedAn_t]
                #else:
                #    qMedAn = qMedAn.append(qMedAn_t)
                
                anhos.append(t)
            
            #print(qMedAn)   
            
            qMedAn = pd.DataFrame(qMedAn, columns=['qmd'],index=anhos)
            qMedAn_dia = pd.DataFrame(columns=['qman'],index=indDia_ts)
            
            for t in indDia_ts:
            
                qMedAn_dia[qMedAn_dia.columns.values[0]][t] = qMedAn[qMedAn.columns.values[0]][t.year]
                #PrAnhoCuenc_dia[PrAnhoCuenc_dia.columns.values[0]][t] = PrAnhoCuenc[PrAnhoCuenc.columns.values[0]][str(t.year)]
            
            
            #print(qMedAn_dia)
            print(qMedAn)
            
            
            
            
            #input('termino una cuenca')

            datos = qMedAn[qMedAn.columns.values[0]].values
            #datos = PrAnhoCuenc['pr (mm/año)'].values
            
            ###···proba_empirica = np.zeros((datos.shape[0],4))
            #proba_empirica = pd.DataFrame(columns=['anho','qMedAn','prob exc','prob ocur'])
            proba_empirica = pd.DataFrame(columns=['anho','qMedAn'])
            #print(proba_empirica)
            #input('probaempirica') 
            ###proba_empirica = np.zeros((datos.shape[0],2))
            
            #datos_ordenados = np.sort(datos[:],axis=0)[::-1]## Ojo con eso, en python3 se requiere que se indique la columna aunque, como en este caso, <datos> es un vector.
            
            proba_empirica['anho'] = anhos
            proba_empirica['qMedAn'] = datos
            
            
            proba_empirica.sort_values('qMedAn', ascending=False, inplace=True)
            
            
            #print(proba_empirica)
            #proba_empirica = proba_empirica.dropna()
            proba_empirica.dropna(inplace=True)
            #print(proba_empirica)
            N = len(proba_empirica)
            proba_empirica.index = range(N)
            vector_nulo = np.zeros((N,1))
            vector_nulo[:] = np.nan
            proba_empirica.insert(len(proba_empirica.columns.values),'prob exc',vector_nulo)
            proba_empirica.insert(len(proba_empirica.columns.values),'prob ocur',vector_nulo)
            #print(proba_empirica)
            #print('estación ', est)
            
            for i in range(N):
                #proba_empirica[i,1] = (i+1) / (N + 1)
                proba_empirica['prob exc'][i] = (i+1) / (N + 1)
                #proba_empirica['prob ocur'][i] = 1 - (i+1) / (N + 1)
                proba_empirica['prob ocur'][i] = 1 - (i+1) / (N + 1)
                
                ###···proba_empirica[i,2] = (N + 1) / (i+1)
                
            #proba_empirica.to_csv(dir_base+'C47_ProbaEmpirica_Pr_f2.csv')
            proba_empirica.to_csv(rutaGuardar + txtDescr+'_AnFrec_Weib.csv')
            print(proba_empirica)
            #proba_2017 = proba_empirica.loc[proba_empirica['anho']==2017]['prob exc'].values[0]
            #print(proba_2017)
            #pr_2017 = proba_empirica.loc[proba_empirica['anho']==2017]['pr anual'].values[0]
            #print(pr_2017)
            
            #ninput('ksldjas')

    ##re    turn indDia_ts


            
            ### ~~~~ ###
            ### La gráficas!!
            ### ~~~~ ###
            params = {'legend.fontsize': 8,
                     'axes.labelsize': '8',
                     'axes.titlesize':'8',
                     'xtick.labelsize':'8',
                     'ytick.labelsize':'8'}

            mpl.rcParams.update(params)

            font = {'family' : 'normal',
                    'weight' : 'regular',
                    'size'   : 8}
            
            mpl.rc('font', **font)
            
            fig = plt.figure(constrained_layout=True,figsize=(10,3))
            gs = fig.add_gridspec(2,1)
            ax = fig.add_subplot(gs[0,0])
            
            #ax = fig.add_subplot(gs[ubicgrafico:ubicgrafico+pasoGraf,:8])
            etiqueta = 'Caudal medio diario ($m^3/s$)'
            ax.plot(indDia_ts, Qdia[est][fechaIni:fechaFin].values, linestyle='-', linewidth=0.7, alpha=0.9, color = 'lightseagreen')#,label=etiqueta)#colores_reg[0])
            tituloEje = 'Q medio diario'+'\n'+ '($m^3/s$)'
            ax.set_ylabel(tituloEje, fontsize=8)
            #ax.legend(loc='upper right', fontsize=8)
            #ax.xaxis.set_major_locator(mdates.YearLocator(1,month=4))
            #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
            ax.grid(b=True, which='major', color='0.60', linestyle='-', linewidth=0.3)
            #ax.grid(b=True, which='minor', color='0.75', linestyle='-',linewidth=0.3)
            ax.set_xlim(fechaIni, fechaFin)
            #yMax = 100
            
            ax.set_xticklabels([])
            #titulo  = 'Río Loa en Desembocadura 02120001-8'
            titulo  = 'Caudal medio diario en estación ' + est
            plt.title(titulo, fontsize=8)
            
            ax = fig.add_subplot(gs[1,0])
            ax.plot(indDia_ts, qMedAn_dia.values, linestyle='--', linewidth=0.7, alpha=0.9, color = 'lightseagreen')#,label=etiqueta)#colores_reg[0])
            #ax.plot(indDia_ts, PrAnhoCuenc_dia[PrAnhoCuenc_dia.columns.values[0]].values, color = 'darkred', linestyle = '--')
            #ax.plot(indDia_ts, PrAnho_SumAcum['pr (mm)'].values, color = 'tomato', linestyle = '-')
            tituloEje = 'Caudal medio anual '+'\n'+ '($m^3/s$)'
            ax.grid(b=True, which='major', color='0.6', linestyle='-', linewidth=0.3)
            ax.set_ylabel(tituloEje, fontsize=8)
            
            
            ax.set_xlim(fechaIni,fechaFin)
            titulo  = 'Caudal medio anual'
            #plt.title(titulo, fontsize=8.5)
            
            #input('alkjshad')
            
            ####···axsec = ax.twinx()
            ####···#axsec.plot(anhos_ts, PrAnhoCuenc[PrAnhoCuenc.columns.values[0]].values, color = 'orange')
            ####···axsec.plot(indDia_ts, PrAnhoCuenc_dia[PrAnhoCuenc_dia.columns.values[0]].values, color = 'tomato', linestyle = '--')
            ####···tituloEje = 'Pr'+'\n'+ '(mm/d)'
            ####···axsec.grid(b=True, which='major', color='0.60', linestyle='-', linewidth=0.3)
            ####···axsec.set_ylabel(tituloEje, fontsize=8, color='tomato')
            
            #plt.savefig(dir_base+'C47_SerieTemp_Pr_f2.png', bbox_inches='tight')
            plt.savefig(rutaGuardar + txtDescr+'_Qmd_SerieTemporal.png',dpi=300,bbox_inches='tight')
            #plt.savefig(dir_base+txtDescr+'_SerieTemp_Pr_f2.png', bbox_inches='tight')
            plt.close()

#otro gr    áfico

#input('    antes de la priobva')    


            fig = plt.figure(constrained_layout=True,figsize=(4.5,4))
            gs = fig.add_gridspec(1,1)
            ax = fig.add_subplot(gs[0,0])
            
            #ax = fig.add_subplot(gs[ubicgrafico:ubicgrafico+pasoGraf,:8])
            #etiqueta = 'Precipitacion diaria media en la cuenca (mm/dia)'
            #ax.plot(todoDato.index.values, todoDato[todoDato.columns.values[4]].values, linestyle='-', linewidth=0.4, alpha=0.9, color = 'darkred',label=etiqueta)#colores_reg[0])
            ax.plot(proba_empirica['prob exc'].values, proba_empirica['qMedAn'].values, linestyle='-', marker='o',markersize=2.5, linewidth=0.7, alpha=0.9, color = 'lightseagreen')#,label=etiqueta)#colores_reg[0])
            
            #anho85 = 1979
            #prAnual85 = 124.17  
            
            #ax.annotate('Año 2017'+'\n'+'Prob. exc = '+str(round(proba_2017*100,0))+'%',
            #            xy=(proba_2017, pr_2017),  # theta, radius
            #            #xy=(0.85, prAnual85),  # theta, radius
            #            xytext=(0.7, 0.455),    # fraction, fraction
            #            textcoords='figure fraction',
            #            arrowprops=dict(facecolor='darkred', arrowstyle='->'),
            #            horizontalalignment='left',
            #            verticalalignment='bottom')
            
            
            
            
            #ax.plot(PrDiaCuenc.index.values, PrDiaCuenc['pr (mm/dia)'].values, linestyle='-', linewidth=0.4, alpha=0.9, color = 'darkred')#,label=etiqueta)#colores_reg[0])
            #tituloEje = 'Caudal medio anual'+'\n'+ '($m^3/s$)'
            tituloEje = 'Caudal medio anual'+' '+ '($m^3/s$)'
            ax.set_ylabel(tituloEje, fontsize=8)
            tituloEje = 'Probabilidad de excedencia'
            ax.set_xlabel(tituloEje, fontsize=8)
            #ax.legend(loc='upper right', fontsize=8)
            #ax.xaxis.set_major_locator(mdates.YearLocator(1,month=4))
            #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
            ax.grid(b=True, which='major', color='0.60', linestyle='-', linewidth=0.3)
            #ax.grid(b=True, which='minor', color='0.75', linestyle='-',linewidth=0.3)
            ax.set_xlim(0, 1)
            #yMax = 100
            
            
            
            #titulo = 'Curva duracion caudal medio anual - Cuenca Río Loa'

            #titulo = txtDescr+'_Est'+est+'_Curva duracion caudal medio anual'
            titulo ='Curva duracion caudal medio anual' + '\n' + txtDescr
            plt.title(titulo, loc='left')
            
            #plt.savefig(dir_base+'C47_CurvaDuracion_Pr_f2.png', bbox_inches='tight')
            plt.savefig(rutaGuardar + txtDescr+'_CurvaDuracion_qMedAn.png', bbox_inches='tight')
            plt.close()



    comentarios.close()

    #input('Qdia max ah?? ?')


def Correlaciones_DF(estadist_DF, rutaGuardar, txtDescr):
	datos_crudos = estadist_DF 
	nEst = len(datos_crudos.columns.values)
	mask = np.zeros((nEst,nEst))
	mask[np.triu_indices_from(mask)] = True
	# Genera una matriz tipo mapa de calor con las correlaciones
	#plt.rcParams['figure.figsize'] = (4.0, 4.0)
	fig = plt.figure(figsize=(3.3,3))
	ax = sns.heatmap(datos_crudos.corr(), mask=mask, annot=True,  cmap='Blues',vmin = 0.0)#, vmax=0.95)
	#ax.grid(b=False, which='major', color='0.60', linestyle='-', linewidth=0.6)
	#ax.grid(b=False, which='minor', color='0.75', linestyle='-',linewidth=0.3)
	#ax.set_xticks()
	#ax.set_yticks()
	#ax.set_ylim((19,0))

	#plt.title('Matriz de correlaciones inter estaciones - Caudal medio mensual', loc='left')
	plt.title('Matriz de coeficientes de correlaciín lineal - Qmd', loc='left', fontsize = 8)
        


	fig.align_labels()
	plt.savefig(rutaGuardar+txtDescr+'_MatrizCoefCorr_Qmd'+'.png', bbox_inches='tight', dpi=300)
	plt.close()



	# Genera la matriz de correlaciones
	axes1 = scatter_matrix(datos_crudos, alpha = 0.7, diagonal='kde', grid=True, figsize=(3,3))
	plt.savefig(rutaGuardar+txtDescr+'_MatrizCorr_Qmd'+'.png', bbox_inches='tight', dpi=300)
	#plt.savefig(directorio+'GráficaCorrel_'+archivo[:-5]+'.png', bbox_inches='tight', dpi=350)
	plt.title('Matriz de correlaciones - Qmd', loc='left', fontsize = 8)
	plt.close()

	#input('pausa')
	# Genera correlaciones cruzadas entre cada par de estaciones presentes en el archivo de datos

	for i in range(len(datos_crudos.columns.values)):
		for j in range(len(datos_crudos.columns.values)):
			if i != j:
				correl = pd.DataFrame()
				correl[datos_crudos.columns.values[j]] = datos_crudos[datos_crudos.columns.values[j]]
				correl[datos_crudos.columns.values[i]] = datos_crudos[datos_crudos.columns.values[i]]
				correl.dropna(inplace=True)
				print('largo de datos en comun -> ',len(correl))
				if len(correl) > 2:
					est0 = correl[correl.columns[0]].values
					est1 = correl[correl.columns[1]].values
					
					est0 = est0.reshape(-1,1)
					est1 = est1.reshape(-1,1)

					#print(est0.shape)

					#Genera el objeto que realiza el ajuste lineal
					regre = linear_model.LinearRegression(fit_intercept=True)
					#Se le indican los datos para la regresión
					regre.fit(est0, est1)

					fig, ax = plt.subplots(figsize=(3,3))
					ax.plot(est0, est1, color='#1f64cc', marker='o', markersize=5, linestyle='None', alpha = 0.6)	
					ax.plot(est0, regre.predict(est0), color='c', linewidth = 0.8, linestyle='--')

					x_max = max(est0) 
					y_max = max(est1)
					#ax5.set_xlim(0.0, 1.05 * caudales_df_maxDia['QmaxDia [m3/s]'].max())
					ax.set_xlim(0.0, 1.05 * x_max)
					ax.set_ylim(0.0, 1.05 * y_max)
					plt.grid(b=True, which='major', color='0.60', linestyle='-', linewidth=0.8)
					plt.grid(b=True, which='minor', color='0.75', linestyle='-',linewidth=0.6)
					#texto_ajuste = 'pendiente = '+ str(round(regre.coef_[0][0],2)) + '\n'	+'intercepto = '+ str(round(regre.intercept_[0],4)) + '\n'  +'R^2 = '+str(round(regre.score(q_medDia_0,q_medDia_1),6))# + '\n'  +'R^2 = '+str(round(regre.score(q_medDia_1,regre.predict(q_medDia_0)),9))+ '\n'  +'R^2 = '+str(round(regre.score(regre.predict(q_medDia_0),q_medDia_1),9)) 
					#if regre.score(est0,est1) > 0.5:
					#texto_ajuste = 'pendiente = '+ str(format(regre.coef_[0][0],"^.2f")) + '\n'	+'intercepto = '+ str(format(regre.intercept_[0],"^.4f")) + '\n'  +'$R^2$ = '+str(format(regre.score(est0,est1),"^.3f"))
					texto_ajuste = '$y(x)$ = '+ str(format(regre.coef_[0][0],"^.2f"))+'$\cdot x + $ ' + str(format(regre.intercept_[0],"^.4f")) + '\n'  +'$R^2$ = '+str(format(regre.score(est0,est1),"^.3f"))
					plt.text(0.05 * x_max, 1.0 * y_max, texto_ajuste, backgroundcolor='w',verticalalignment='top')

					#plt.xlabel('Caudal medio mensual [m3/s] - Estación ' + str(correl.columns[0]))
					#plt.ylabel('Caudal medio mensual [m3/s] - Estación ' + str(correl.columns[1]))
					#plt.title('Correlación QmedMensual Est. '+ str(correl.columns.values[1]) + ' v/s QmedMensual Est. ' + str(correl.columns.values[0]) , loc='left', fontsize=10)
					#plt.savefig(directorio+'Correl_Qmm_'+correl.columns[1]+'-'+correl.columns[0]+'.png', bbox_inches='tight')

					plt.xlabel('Caudal medio diario [m3/s] - Estación ' + str(correl.columns[0]))
					plt.ylabel('Caudal medio diario [m3/s] - Estación ' + str(correl.columns[1]))
					plt.title('Correlación QmedDia Est. '+'\n' + str(correl.columns.values[1]) + ' v/s' +  'Est. ' + str(correl.columns.values[0]) , loc='left', fontsize=8)
					plt.savefig(rutaGuardar+txtDescr+'Correl_Qmd_'+correl.columns[1]+'-'+correl.columns[0]+'.png', bbox_inches='tight')
					#plt.show()
					plt.close()
				else:
					print('Las estaciones ',correl.columns[1],' y ',correl.columns[0],' no tienen datos en común para correlacionar')



def tablaLatex(DataFrame,rutaGuardar,txtDescr):
    
    nombreArchivo = txtDescr + '_TablaCalidad'+'.tex'
    archivoTex = open(rutaGuardar+nombreArchivo, 'w+') 

    baseTabla = open(rutaGuardar+'_baseTabla_v1.tex')
    
    encab = baseTabla.read()
    #encab = '\documentclass{article}'+'\n'+'\usepackage[paperheight=3in,paperwidth=9in,heightrounded, margin=0.2in]{geometry}'+'\n'+'\usepackage{booktabs}'+'\n'#+'\begin{document}'

    #print(encab)
    archivoTex.write(encab)


    linEntr = DataFrame.to_latex()
    ##linEntr = calidad.to_latex()
    archivoTex.write(linEntr)

    cierreTabla = open(rutaGuardar+'_cierreTabla_v1.tex')
    
    cierre = cierreTabla.read()
    archivoTex.write(cierre)
    
    archivoTex.close()
    
   
    comando = 'pdflatex '+rutaGuardar+nombreArchivo
    print(comando)
    input('ante comand') 
    os.system(comando)
    input('encab') 








