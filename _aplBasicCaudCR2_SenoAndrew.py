# -*- coding: utf-8 -*-
#!/usr/bin/env python


import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 						# matplotlib 3.0.2	for i in range(len(f)):
from BasicCaudCR2_v200 import importa_Q_CR2, Correlaciones_DF, creaCarpeta, AnFrec_weibull, calidadEstadistica

#  * #####
# ** ## Antecdentes de entrada
#  * #####

rutArchivos = '/home/simonfangos/Datos/CR2_estadistica/cr2_qfkxDaily_2018/'

estaciones = [
#'12280004',
#'12287002',
#'12283003',
##'12284002', Contenida
#'12288004',
#'12285001', ## Rio Chorrillos Tres Pasos Ruta N 9, 101 km2,
##'12284005', Contenida
#'12288002',
##'12287001', ## Rio Grey Antes Junta Serrano, 865 km2, Contenida
##'12284007', ## Rio Las Chinas Antes Desague Del Toro, 3939 km2, Contenida
##'12284006', Contenida
#'12284004',
#'12280001',
'12280002', ## Rio Paine En Parque Nacional 2, Contenida
#'12291001',
#'12286002',
#'12289003',
#'12289002', ## Río Serrano en Desagüe Lago Toro, Contenida
'12289001', ## Río Serrano en Desembocadura, 8583 km2
#'12288003',
#'12285003',
#'12284003', ## Contenida
]

fechaIni = '01/01/1980'
fechaFin = '31/12/2018'

intento = 'SenoAndrew_v2'
rutaGuardar = '/home/simonfangos/Documentos/01.Empleo/04INH/PlanHidrico/5oferta/'
rutaGuardar = rutaGuardar+str(intento)+'/'

#  * ##### 
# ** ## Aplicación 
#  * #####

#· crea una carpeta para guardar las cosas
creaCarpeta(rutaGuardar)

txtDescr = intento
#· importa caudales CR2 para las fechas y estaciones indicadas
Qdia = importa_Q_CR2(rutArchivos, rutaGuardar, fechaIni,fechaFin, estaciones, txtDescr)    

#· realiza un filtro de valores específicos (caso algunas estaciones australes)
###····Qdia = Qdia.replace(1.0,np.nan)
###····Qdia = Qdia.replace(2.0,np.nan)
###····Qdia = Qdia.replace(3.0,np.nan)

#· realiza correlaciones cruzadas entre las estciones seleccionadas
Correlaciones_DF(Qdia, rutaGuardar, txtDescr)

#· realiza una revisión de la extensión e identificación de gaps
calidadEstadistica(Qdia, rutaGuardar,txtDescr)

#· realiza un análisis de frecuenca a caudales medios anuales
AnFrec_weibull(Qdia[estaciones],rutaGuardar)

print('Exito')
print('Los resultados deben estar guardados en:')
print(rutaGuardar)




