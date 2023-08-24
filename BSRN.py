#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:16:02 2023

@author: dario
"""

import pandas as pd
import numpy as np
import math


class BSRN:
    def __init__(self, df, c1,c2, c3, Prs):
        TSI = 1361
        E = TSI * df.E0
        df['TZgrad'] = df.TZ.apply(math.degrees)
        #filtro diurno
        df['f0'] = df.CTZ>0
        
        #filtro basado en valores físicamente posibles
        df['f1_GHI'] = (df.GHI > -4) & (df.GHI < 1.5 * E * df.CTZ**1.2 + 100)
        df['f1_DHI'] = (df.DHI > -4) & (df.DHI < 0.95* E * df.CTZ**1.2 + 50)
        df['f1_DNI'] = (df.DNI > -4) & (df.DNI < E)
            
        #filtro basado en valores extremandamente raros
        df['f2_GHI'] = (df.GHI > -2) & (df.GHI < 1.2 * E * df.CTZ**1.2 + 50)
        df['f2_DHI'] = (df.DHI > -2) & (df.DHI < 0.75* E * df.CTZ**1.2 + 30)
        df['f2_DNI'] = (df.DNI > -2) & (df.DNI < 0.95* E * df.CTZ**0.2 + 10)


        #filtro basado en la relacion de clausura
        df['r1'] = abs(100 * (df.DHI + df.DNI*df.CTZ - df.GHI) / df.GHI)
        df['f3_a'] = np.where((df.TZgrad<75), (df.GHI>50) & (df.r1<8) ,True )
        df['f3_b'] = np.where((df.TZgrad>75) & (df.TZgrad<93), (df.GHI>50)&(df.r1<15) ,True )

       
        df['f3cd'] = np.where(df.GHI<50, True, np.where(
            df.TZgrad<75, (df.DHI / df.GHI) < 1.05, (df.DHI / df.GHI) < 1.10))
        
        
        #df['f4'] = np.where((df.DHI>50) & ((df.GHI/df.GHIargp)>0.85), (df.DHI/df.GHI) < 0.85 ,True )
        
        #filtro basado en los límites climatológicos
        df['f4_GHI'] = (df.GHI > 0) & (df.GHI < c1 * E * df.CTZ**1.2 + 50)
        df['f4_DHI'] = (df.DHI > 0) & (df.DHI < c2 * E * df.CTZ**1.2 + 30)
        df['f4_DNI'] = (df.DNI > 0) & (df.DNI < c3 * E * df.CTZ**0.2 + 10)
        
        
        #filtro basado en comparaciones climatológicas
        df['f5_a'] = np.where((df.DHI>50) & ((df.GHI/df.GHIargp)>0.85), (df.DHI/df.GHI) < 0.85 ,True )
        lim = 209.3*df.CTZ - 708.3*df.CTZ**2 + 1128.7*df.CTZ**3 - 911.2*df.CTZ**4 + 287.85*df.CTZ**5 + 0.046725*df.CTZ*Prs
        df['f5_b'] = np.where((df.GHI>50) & ((df.DHI/df.GHI)<0.8), df.DHI > lim - 1 ,True )
        
        df['Tracker'] = np.where( (abs(df.DHI - df.GHI)<10) | (df.DHI>700), False , True )
        
        
        
# ap = pd.read_csv('abrapampa.csv')

# ap = ap.take([2,3,7,11], axis=1)
# ap.columns = ['Fecha', 'GHI', 'DNI', 'DHI']

# ap['Fecha'] = ap.Fecha.str.replace( '\'' , '')
# ap['Fecha'] = ap.Fecha.str.strip()
# ap['Fecha'] = pd.to_datetime(ap.Fecha, format='%d/%m/%Y %H:%M')
# ap = ap.sort_values(by=['Fecha'])


# ap.Fecha.max()


# Fechas = pd.date_range(
#             start=ap.Fecha.min(), 
#             end=ap.Fecha.max(), 
#             freq="1 min")


# esperados = Fechas[(Fechas.hour < 4) | (Fechas.hour >= 7)]



# ap = (ap.set_index('Fecha')
#       .reindex(Fechas)
#       .rename_axis(['Fecha'])
#       #.fillna(0)
#       .reset_index())


# import Geo
# dfGeoAP = Geo.Geo(
#          range_dates=ap['Fecha'],
#          lat=-22.80205, 
#          long=-65.82436, 
#          gmt=-3, 
#          beta=0,
#          alt=3459).df



# mergeap = pd.merge(ap,dfGeoAP)




# mergeap['GHI'] = mergeap.GHI.shift(-180)
# mergeap['DHI'] = mergeap.DHI.shift(-180)
# mergeap['DNI'] = mergeap.DNI.shift(-180)



# mergeap.to_csv('abrapampa_merge.csv', index=False)

# mergeap.dropna(inplace=True)




# mergeap = pd.read_csv('abrapampa_merge.csv')
# mergeap.dropna(inplace=True)
# BSRN(mergeap)



# #mergeap = pd.read_csv('abrapampa_merge.csv')
# mergeap.columns
# orig = len(mergeap)

# plt.plot(mergeap.CTZ, mergeap.GHI, 'o', markersize=0.5, label="GHI")
# plt.title("GHI Abra Pampa - Medidas originales ")
# plt.text(-0.95, 260, f'Datos: {len(mergeap)}', fontsize = 12)
# plt.text(-0.95, 160, f'Max = {mergeap.GHI.max()}wm²', fontsize = 12)
# plt.text(-0.95, 60, f'Min = {mergeap.GHI.min()}wm²', fontsize = 12)
# plt.legend()
# plt.xlabel("CTZ")
# plt.ylabel("GHI")
# plt.show()

# plt.plot(mergeap.CTZ, mergeap.DHI, 'o', markersize=0.5, label="DHI")
# plt.title("DHI Abra Pampa - Medidas originales ")
# plt.text(-0.95, 260, f'Datos: {len(mergeap)}', fontsize = 12)
# plt.text(-0.95, 160, f'Max = {mergeap.DHI.max()}wm²', fontsize = 12)
# plt.text(-0.95, 60, f'Min = {mergeap.DHI.min()}wm²', fontsize = 12)
# plt.legend()
# plt.xlabel("CTZ")
# plt.ylabel("DHI")
# plt.show()

# plt.plot(mergeap.CTZ, mergeap.DNI, 'o', markersize=0.5, label="DNI")
# plt.title("DNI Abra Pampa - Medidas originales ")
# plt.text(-0.95, 260, f'Datos: {len(mergeap)}', fontsize = 12)
# plt.text(-0.95, 160, f'Max = {mergeap.DNI.max()}wm²', fontsize = 12)
# plt.text(-0.95, 60, f'Min = {mergeap.DNI.min()}wm²', fontsize = 12)
# plt.legend()
# plt.xlabel("CTZ")
# plt.ylabel("DNI")
# plt.show()




# mergeap.columns


# mergeap = mergeap[mergeap.f0]

# filtrados = orig - len(mergeap)


# plt.plot(mergeap.CTZ, mergeap.GHI, 'o', markersize=0.5, label="GHI")
# plt.title("GHI Abra Pampa - F0 : CTZ>0 ")
# plt.text(0.01, 1500, f'Datos: {len(mergeap)} ', fontsize = 12)
# #plt.text(0.01, 1400, f'Filtrados: {filtrados} --> {filtrados/orig * 100:.2f} % ', fontsize = 12)
# plt.text(0.01, 1100, f'Max = {mergeap.GHI.max()}wm²', fontsize = 12)
# plt.text(0.01, 1000, f'Min = {mergeap.GHI.min()}wm²', fontsize = 12)
# plt.xlim([mergeap.CTZ.min(), mergeap.CTZ.max()])
# plt.ylim([mergeap.GHI.min(), mergeap.GHI.max()])
# plt.legend()
# plt.xlabel("CTZ")
# plt.ylabel("GHI")
# plt.show()


# plt.plot(mergeap.CTZ, mergeap.DHI, 'o', markersize=0.5, label="DHI")
# plt.title("DHI Abra Pampa - F0 : CTZ>0 ")
# plt.text(0.01, 1500, f'Datos: {len(mergeap)} ', fontsize = 12)
# #plt.text(0.01, 1400, f'Filtrados: {filtrados} --> {filtrados/orig * 100:.2f} % ', fontsize = 12)
# plt.text(0.01, 1100, f'Max = {mergeap.DHI.max()}wm²', fontsize = 12)
# plt.text(0.01, 1000, f'Min = {mergeap.DHI.min()}wm²', fontsize = 12)
# plt.xlim([mergeap.CTZ.min(), mergeap.CTZ.max()])
# plt.ylim([mergeap.DHI.min(), mergeap.DHI.max()])
# plt.legend()
# plt.xlabel("CTZ")
# plt.ylabel("DHI")
# plt.show()

# plt.plot(mergeap.CTZ, mergeap.DNI, 'o', markersize=0.5, label="DNI")
# plt.title("DNI Abra Pampa - F0 : CTZ>0 ")
# plt.text(0.01, 1000, f'Datos: {len(mergeap)} ', fontsize = 12)
# #plt.text(0.01, 900, f'Filtrados: {filtrados} --> {filtrados/orig * 100:.2f} % ', fontsize = 12)
# plt.text(0.01, 700, f'Max = {mergeap.DNI.max()}wm²', fontsize = 12)
# plt.text(0.01, 600, f'Min = {mergeap.DNI.min()}wm²', fontsize = 12)
# plt.xlim([mergeap.CTZ.min(), mergeap.CTZ.max()])
# plt.ylim([mergeap.DNI.min(), mergeap.DNI.max()])
# plt.legend()
# plt.xlabel("CTZ")
# plt.ylabel("DNI")
# plt.show()





# """
# Aplica el filtro F1 y F2, 
# de los límites Fisc. Posibles y Ext. Raros para GHI
# """
# TSI = 1361
# E = TSI * mergeap.E0
# mergeap['TZg'] = mergeap.TZ.apply(math.degrees) 

# plt.title("GHI Abra Pampa - F1 y F2 ")
# plt.plot(mergeap.TZg, mergeap.GHI, 'o', markersize=0.1, label="GHI", color="black")
# plt.plot(mergeap.TZg, 1.5 * E * mergeap.CTZ**1.2 + 100, 'o', color="red", markersize=0.1, label="Fisc. Posible")
# plt.plot(mergeap.TZg, 1.2 * E * mergeap.CTZ**1.2 + 50, 'o', color="blue", markersize=0.1, label="Ext. Raros")
# plt.legend(markerscale=60)
# plt.xlabel("TZ °")
# plt.ylabel("GHI wm²")
# plt.show()


# """
# Aplica el filtro F1 y F2, 
# de los límites Fisc. Posibles y Ext. Raros para DHI
# """


# plt.title("DHI Abra Pampa - F1 y F2 ")
# plt.plot(mergeap.TZg, mergeap.DHI, 'o', markersize=0.1, label="DHI", color="black")
# plt.plot(mergeap.TZg, 0.95 * E * mergeap.CTZ**1.2 + 50, 'o', color="red", markersize=0.1, label="Fisc. Posible")
# plt.plot(mergeap.TZg, 0.75 * E * mergeap.CTZ**1.2 + 30, 'o', color="blue", markersize=0.1, label="Ext. Raros")
# plt.legend(markerscale=60)
# plt.xlabel("TZ°")
# plt.ylabel("DHI wm²")
# plt.show()



# """
# Aplica el filtro F1 y F2, 
# de los límites Fisc. Posibles y Ext. Raros para DNI
# """


# plt.title("DNI Abra Pampa - F1 y F2 ")
# plt.plot(mergeap.TZg, mergeap.DNI, 'o', markersize=0.1, label="DNI", color="black")
# plt.plot(mergeap.TZg, E , 'o', color="red", markersize=0.1, label="Fisc. Posible")
# plt.plot(mergeap.TZg, 0.95 * E * mergeap.CTZ**0.2 + 10, 'o', color="blue", markersize=0.1, label="Ext. Raros")
# plt.legend(markerscale=60)
# plt.xlabel("TZ°")
# plt.ylabel("DNI wm²")
# plt.show()


# """"Comparaciones con la relación de clausura"""
# """Comparación A"""
# mergeap['Closr'] = (mergeap.DHI+ mergeap.DNI * mergeap.CTZ) / mergeap.GHI
# mergeap['limitesupclousure'] = np.where(mergeap.TZg<75, 1.08, 1.15 )
# mergeap['limiteinfclousure'] = np.where(mergeap.TZg<75, 0.92, 0.85 )

# plt.title("Closr Abra Pampa - F3 - Closr  ")
# plt.plot(mergeap[mergeap.GHI>50].TZg, mergeap[mergeap.GHI>50].Closr, 'o', markersize=0.1, label="Closr", color="black")
# plt.plot(mergeap[mergeap.GHI>50].TZg, mergeap[mergeap.GHI>50].limitesupclousure, 'o', markersize=0.1, label="", color="red")
# plt.xlabel("TZ°")
# plt.ylabel("Closr wm²")
# plt.show()

# """Comparación B"""

# mergeap['limiteF3b'] = np.where(mergeap.TZg<75, 1.05, 1.10 )


# plt.title("Closr Abra Pampa - F3 DHI/GHI ")
# plt.plot(mergeap[mergeap.GHI>50].TZg, mergeap[mergeap.GHI>50].DHI / mergeap[mergeap.GHI>50].GHI, 'o', markersize=0.1, label="DHI/GHI", color="black")
# plt.plot(mergeap[mergeap.GHI>50].TZg, mergeap[mergeap.GHI>50].limiteF3b, 'o', markersize=0.1, label="DHI/GHI", color="red")

# #plt.plot(mergeap[mergeap.GHI>50].TZg, mergeap[mergeap.GHI>50].limitesupclousure, 'o', markersize=0.1, label="", color="red")
# plt.xlabel("TZ°")
# plt.ylabel("Closr wm²")
# plt.legend(markerscale=60)
# plt.show()


# mergeap['kt'] = mergeap.GHI / mergeap.GHIargp

# """
# Comparacion climatológica Clr
# """

# mask = [0.85 for i in (mergeap[ (mergeap.kt>0.85) & (mergeap.DHI>50)]).Fecha]
# plt.plot(mergeap[ (mergeap.kt>0.85) & (mergeap.DHI>50)].TZg, mergeap[ (mergeap.kt>0.85) & (mergeap.DHI>50)].DHI/mergeap[ (mergeap.kt>0.85) & (mergeap.DHI>50)].GHI, 'o',markersize=0.5, color="black", label="DHI/GHI"  )
# plt.plot(mergeap[ (mergeap.kt>0.85) & (mergeap.DHI>50)].TZg, mask, 'o',markersize=0.5, color="red"  )


# """
# Error en el tracker
# """

# mergeap['dif_GHIDHI'] = mergeap.DHI - mergeap.GHI / 10
# mergeap['max_GHIDHI'] = 1/(mergeap.DHI - mergeap.GHI / 10).max()

# plt.plot(mergeap.TZg, mergeap.dif_GHIDHI/mergeap.dif_GHIDHI.max(), 'o', markersize=0.5, color="black", label="DIF |DHI-GHI|")
# plt.plot(mergeap.TZg, mergeap.max_GHIDHI, 'o', markersize=0.5, color="red", label="DIF |DHI-GHI|")
# plt.plot(mergeap.TZg, -mergeap.max_GHIDHI, 'o', markersize=0.5, color="red", label="DIF |DHI-GHI|")






# dffil = mergeap[mergeap.f0 & mergeap.f1_GHI & mergeap.f1_DHI & mergeap.f1_DNI]

# filtrados = orig - len(mergeap)



# plt.plot(mergeap.CTZ, 1.5 * E * mergeap.CTZ**1.2 + 100, 'o', color="red", markersize=0.05)
# plt.plot(mergeap.CTZ, mergeap.GHI, 'o', markersize=0.5, label="GHI")

# plt.text(0.01, 1500, f'Datos: {len(mergeap)} ', fontsize = 12)
# plt.text(0.01, 1400, f'Filtrados: {filtrados} --> {filtrados/orig * 100:.2f} % ', fontsize = 12)
# plt.text(0.01, 1100, f'Max = {mergeap.GHI.max()}wm²', fontsize = 12)
# plt.text(0.01, 1000, f'Min = {mergeap.GHI.min()}wm²', fontsize = 12)
# plt.xlim([mergeap.CTZ.min(), mergeap.CTZ.max()])
# plt.ylim([mergeap.GHI.min(), mergeap.GHI.max()])
# plt.legend()
# plt.xlabel("CTZ")
# plt.ylabel("GHI")
# plt.show()




# plt.plot(mergeap.CTZ, 0.95 * E * mergeap.CTZ**1.2 + 50, 'o', color="red", markersize=0.05)
# plt.plot(mergeap.CTZ, mergeap.DHI, 'o', markersize=0.5, label="DHI")
# plt.title("DHI Abra Pampa - F1 : CTZ>0 ")
# plt.text(0.01, 1000, f'Datos: {len(mergeap)} ', fontsize = 12)
# plt.text(0.01, 900, f'Filtrados: {filtrados} --> {filtrados/orig * 100:.2f} % ', fontsize = 12)
# plt.text(0.01, 700, f'Max = {mergeap.DHI.max()}wm²', fontsize = 12)
# plt.text(0.01, 600, f'Min = {mergeap.DHI.min()}wm²', fontsize = 12)
# plt.xlim([mergeap.CTZ.min(), mergeap.CTZ.max()])
# plt.ylim([mergeap.DHI.min(), mergeap.DHI.max()])
# plt.legend()
# plt.xlabel("CTZ")
# plt.ylabel("DHI")
# plt.show()



# plt.plot(mergeap.CTZ, TSI * mergeap.E0, 'o', color="red", markersize=0.05)
# plt.plot(mergeap.CTZ, mergeap.DNI, 'o', markersize=0.5, label="DNI")
# plt.title("DNI Abra Pampa - F1 : CTZ>0 ")
# plt.text(0.01, 1000, f'Datos: {len(mergeap)} ', fontsize = 12)
# plt.text(0.01, 900, f'Filtrados: {filtrados} --> {filtrados/orig * 100:.2f} % ', fontsize = 12)
# plt.text(0.01, 700, f'Max = {mergeap.DNI.max()}wm²', fontsize = 12)
# plt.text(0.01, 600, f'Min = {mergeap.DNI.min()}wm²', fontsize = 12)
# #plt.xlim([mergeap.CTZ.min(), mergeap.CTZ.max()])
# #plt.ylim([mergeap.DNI.min(), mergeap.DNI.max()])
# plt.legend()
# plt.xlabel("CTZ")
# plt.ylabel("DNI")
# plt.show()



# """
# Aplica el filtro F2, de los límites Ext. Raros
# """
# orig = len(mergeap)
# mergeap = mergeap[
#         mergeap.f0 & 
#         mergeap.f1_GHI & 
#         mergeap.f1_DHI & 
#         mergeap.f1_DNI &        
#         mergeap.f2_GHI & 
#         mergeap.f2_DHI & 
#         mergeap.f2_DNI ]

# filtrados = orig - len(mergeap)
# E = TSI * mergeap.E0
# plt.plot(mergeap.CTZ, 1.5 * E * mergeap.CTZ**1.2 + 50, 'o', color="red", markersize=0.05)
# plt.plot(mergeap.CTZ, mergeap.GHI, 'o', markersize=0.5, label="GHI")
# plt.title("GHI Abra Pampa - F2 : CTZ>0 ")
# plt.text(0.01, 1500, f'Datos: {len(mergeap)} ', fontsize = 12)
# plt.text(0.01, 1400, f'Filtrados: {filtrados} --> {filtrados/orig * 100:.2f} % ', fontsize = 12)
# plt.text(0.01, 1100, f'Max = {mergeap.GHI.max()}wm²', fontsize = 12)
# plt.text(0.01, 1000, f'Min = {mergeap.GHI.min()}wm²', fontsize = 12)
# plt.xlim([mergeap.CTZ.min(), mergeap.CTZ.max()])
# plt.ylim([mergeap.GHI.min(), mergeap.GHI.max()])
# plt.legend()
# plt.xlabel("CTZ")
# plt.ylabel("GHI")
# plt.show()




# plt.plot(mergeap.CTZ, 0.75 * E * mergeap.CTZ**1.2 + 30, 'o', color="red", markersize=0.05)
# plt.plot(mergeap.CTZ, mergeap.DHI, 'o', markersize=0.5, label="DHI")
# plt.title("DHI Abra Pampa - F2 : CTZ>0 ")
# plt.text(0.01, 1000, f'Datos: {len(mergeap)} ', fontsize = 12)
# plt.text(0.01, 900, f'Filtrados: {filtrados} --> {filtrados/orig * 100:.2f} % ', fontsize = 12)
# plt.text(0.01, 700, f'Max = {mergeap.DHI.max()}wm²', fontsize = 12)
# plt.text(0.01, 600, f'Min = {mergeap.DHI.min()}wm²', fontsize = 12)
# plt.xlim([mergeap.CTZ.min(), mergeap.CTZ.max()])
# plt.ylim([mergeap.DHI.min(), mergeap.DHI.max()])
# plt.legend()
# plt.xlabel("CTZ")
# plt.ylabel("DHI")
# plt.show()



# plt.plot(mergeap.CTZ, 0.95* E * mergeap.CTZ**0.2 + 10, 'o', color="red", markersize=0.05)
# plt.plot(mergeap.CTZ, mergeap.DNI, 'o', markersize=0.5, label="DNI")
# plt.title("DNI Abra Pampa - F2 : CTZ>0 ")
# plt.text(0.01, 1000, f'Datos: {len(mergeap)} ', fontsize = 12)
# plt.text(0.01, 900, f'Filtrados: {filtrados} --> {filtrados/orig * 100:.2f} % ', fontsize = 12)
# plt.text(0.01, 700, f'Max = {mergeap.DNI.max()}wm²', fontsize = 12)
# plt.text(0.01, 600, f'Min = {mergeap.DNI.min()}wm²', fontsize = 12)
# #plt.xlim([mergeap.CTZ.min(), mergeap.CTZ.max()])
# #plt.ylim([mergeap.DNI.min(), mergeap.DNI.max()])
# plt.legend()
# plt.xlabel("CTZ")
# plt.ylabel("DNI")
# plt.show()



# orig = len(mergeap)
# mergeap = mergeap[
#         mergeap.f0 & 
#         mergeap.f1_GHI & 
#         mergeap.f1_DHI & 
#         mergeap.f1_DNI &        
#         mergeap.f2_GHI & 
#         mergeap.f2_DHI & 
#         mergeap.f2_DNI &
#         mergeap.f3_a &
#         mergeap.f3_b ]
# filtrados = orig - len(mergeap)
# E = TSI * mergeap.E0



# mergeap['TZg'] = mergeap.TZ.apply(math.degrees) 



# limitesup =  (mergeap.DHI +mergeap.DNI*mergeap.CTZ)/mergeap.GHI*1.08
# limiteinf =  (mergeap.DHI +mergeap.DNI*mergeap.CTZ)/mergeap.GHI*1.15

# mergeap['limitesupclousure'] = np.where(mergeap.TZg<75, 1.08, 1.15 )
# mergeap['limiteinfclousure'] = np.where(mergeap.TZg<75, 0.92, 0.85 )


# plt.plot(mergeap.TZ, (mergeap.DHI +mergeap.DNI*mergeap.CTZ)/mergeap.GHI, 'o', markersize=0.05)
# plt.plot(mergeap.TZ, mergeap.limiteinfclousure, 'o',markersize=0.5, color="red")
# plt.plot(mergeap.TZ, mergeap.limitesupclousure, 'o',markersize=0.5, color="red")





# orig = len(mergeap)
# mergeap = mergeap[
#         mergeap.f0 & 
#         mergeap.f1_GHI & 
#         mergeap.f1_DHI & 
#         mergeap.f1_DNI &        
#         mergeap.f2_GHI & 
#         mergeap.f2_DHI & 
#         mergeap.f2_DNI &
#         mergeap.f3_a &
#         mergeap.f3_b & 
#         mergeap.f3_c &
#         mergeap.f3_d]
# filtrados = orig - len(mergeap)
# E = TSI * mergeap.E0


# limitesf3cd = np.where(mergeap.TZg<75, 1.05, 1.10)

# mergeap['limitesf3cd'] = limitesf3cd
# mergeap['f3ab'] = np.where(mergeap.DHI/mergeap.GHI< limitesf3cd, True, False)

# mergeap = mergeap[mergeap.f3ab]
# # mergeap['limitesf3cd'] = np.where(mergeap.GHI.min()<50, np.NAN, mergeap['limitesf3cd'] )


# plt.plot(mergeap.TZg, mergeap.DHI/mergeap.GHI, 'o', markersize=0.5 )
# plt.plot(mergeap.TZg, limitesf3cd , 'o', markersize=0.5, color="red" )

# plt.plot(mergeap.TZg, mergeap.GHI,'o', c="black", markersize=0.15)



# """Error en el tracker A"""

# mergeap['limite_tracker_a'] = np.where( (mergeap.DHI>50)  & ((mergeap.GHI/mergeap.GHIargp)>0.85), 0.85, np.nan  )
# plt.plot(mergeap.TZg, mergeap.DHI/ mergeap.GHI, 'o', markersize=0.5)
# plt.plot(mergeap.TZg, mergeap['limite_tracker_a'], color='red')

# """error en el tracker b"""

# mergeap['uno'] = 1
# plt.plot(mergeap.CTZ, abs(mergeap.DHI - mergeap.GHI) / 10, 'o', markersize=0.5)


# plt.plot( mergeap.CTZ ,mergeap.uno, 'o', markersize=0.5, color="red")

# mergeap['sete'] = 700
# plt.plot( mergeap.CTZ ,mergeap.DHI, 'o', markersize=0.5)
# plt.plot( mergeap.CTZ , mergeap.sete, 'o', markersize=0.5, color="red")




# mergeap.columns


# filtrados = mergeap[
#         mergeap.f0 & 
#         mergeap.f1_GHI & 
#         mergeap.f1_DHI & 
#         mergeap.f1_DNI &        
#         mergeap.f2_GHI & 
#         mergeap.f2_DHI & 
#         mergeap.f2_DNI &
#         mergeap.f3_a &
#         mergeap.f3_b & 
#         mergeap.f3_c &
#         mergeap.f3_d &
#         mergeap.f3cd & 
#         mergeap.f4 &
#         mergeap.Tracker]

# filtrados['TZg'] = filtrados.TZ.apply(math.degrees)

# E = 1361 * filtrados.Fn 



# plt.plot(
#     filtrados.TZg, filtrados.GHI, 'o', 
#     color="black",markersize=0.5, label="GHI Filtrada")
# plt.plot(filtrados.TZg, 1.5 * E * filtrados.CTZ**1.2 + 100, 'o', color="red", markersize=0.1, label="Fisc. Posible")
# plt.plot(filtrados.TZg, 1.2 * E * filtrados.CTZ**1.2 + 50, 'o', color="blue", markersize=0.1, label="Ext. Raros")


# plt.plot(
#     filtrados.TZg, filtrados.DHI, 'o', 
#     color="black",markersize=0.2, label="DHI Filtrada")
# plt.plot(filtrados.TZg, 0.95 * E * filtrados.CTZ**1.2 + 50, 'o', color="red", markersize=0.1, label="Fisc. Posible")
# plt.plot(filtrados.TZg, 0.75 * E * filtrados.CTZ**1.2 + 30, 'o', color="blue", markersize=0.1, label="Ext. Raros")
# plt.legend(markerscale=60)


# plt.plot(
#     filtrados.CTZ, filtrados.DNI, 'o', 
#     color="black",markersize=0.1, label="DNI Filtrada")
# plt.plot(filtrados.CTZ,   E, 'o', color="red", markersize=0.1, label="Fisc. Posible")
# plt.plot(mergeap.CTZ, 0.95* E * filtrados.CTZ**0.2 + 10, 'o', color="blue", markersize=0.1, label="Ext. Raros")
# plt.legend(markerscale=60)




# plt.title("Closr Abra Pampa - F3 - Closr  ")
# plt.plot(filtrados[filtrados.GHI>50].TZg, filtrados[filtrados.GHI>50].Closr, 'o', markersize=0.1, label="Closr", color="black")
# plt.plot(filtrados[filtrados.GHI>50].TZg, filtrados[filtrados.GHI>50].limitesupclousure, 'o', markersize=0.1, label="", color="red")
# plt.plot(filtrados[filtrados.GHI>50].TZg, filtrados[filtrados.GHI>50].limiteinfclousure, 'o', markersize=0.1, label="", color="red")
# plt.xlabel("TZ°")
# plt.ylabel("Closr wm²")
# plt.show()





# filtrados['LimiteF3cd']= np.where(filtrados.TZg<75, 1.05, 1.10)
# f3c = filtrados[(filtrados.GHI>50) & (filtrados.TZgrad<75) & ((filtrados.DHI / filtrados.GHI) > 1.05)]
# f3d = filtrados[(filtrados.GHI>50) & (filtrados.TZgrad>75) & ((filtrados.DHI / filtrados.GHI) > 1.10)]


# plt.title("Closr Abra Pampa - F3 DHI/GHI ")
# plt.plot(filtrados[filtrados.GHI>50].TZg, filtrados[filtrados.GHI>50].DHI / filtrados[filtrados.GHI>50].GHI, 'o', markersize=0.1, label="DHI/GHI", color="black")
# plt.plot(filtrados[filtrados.GHI>50].TZg, filtrados[filtrados.GHI>50].LimiteF3cd, 'o', markersize=0.1, label="DHI/GHI", color="red")
# plt.plot(f3c[f3c.GHI>50].TZg, f3c[f3c.GHI>50].DHI / f3c[f3c.GHI>50].GHI, 'o', markersize=0.1, label="DHI/GHI", color="red")
# plt.plot(f3d[f3d.GHI>50].TZg, f3d[f3d.GHI>50].DHI / f3d[f3d.GHI>50].GHI, 'o', markersize=0.1, label="DHI/GHI", color="red")
# #plt.plot(mergeap[mergeap.GHI>50].TZg, mergeap[mergeap.GHI>50].limitesupclousure, 'o', markersize=0.1, label="", color="red")
# plt.xlabel("TZ°")
# plt.ylabel("Closr wm²")
# plt.legend(markerscale=60)
# plt.show()






# """
# Error en el tracker
# """

# filtrados['dif_GHIDHI'] = abs(filtrados.DHI - filtrados.GHI) / 10
# filtrados['dif_GHIDHI'] = filtrados['dif_GHIDHI'] / filtrados['dif_GHIDHI'].max()
# filtrados['max_GHIDHI'] = 1/(mergeap.DHI - mergeap.GHI / 10).max()

# plt.plot(filtrados.TZg, filtrados['dif_GHIDHI'], 'o', markersize=0.5, color="black", label="DIF |DHI-GHI|")
# plt.plot(filtrados.TZg, filtrados.max_GHIDHI, 'o', markersize=0.5, color="red", label="DIF |DHI-GHI|")
# plt.plot(filtrados.TZg, -filtrados.max_GHIDHI, 'o', markersize=0.5, color="red", label="DIF |DHI-GHI|")



# """
# Limites climatologicos
# """
# c1=1.015
# c2 =0.70
# c3 = 0.90
# #filtrados['Cli1'] = filtrados.GHI>0 & filtrados.GHI<c1*E*filtrados.CTZ**1.2 + 50

# filtrados = filtrados.sort_values(by=['CTZ'])

# plt.plot(
#     filtrados.TZg, filtrados.GHI, 'o', 
#     color="black",markersize=0.05, label="GHI Filtrada")
# #plt.plot(filtrados.TZg, 1.5 * E * filtrados.CTZ**1.2 + 100, 'o', color="red", markersize=0.001, label="Fisc. Posible")
# #plt.plot(filtrados.TZg, 1.2 * E * filtrados.CTZ**1.2 + 50, 'o', color="blue", markersize=0.001, label="Ext. Raros")
# plt.plot(filtrados.TZg, c1 * E * filtrados.CTZ**1.2 + 50, 'o', color="blue", markersize=0.003, label="Lim. Climatológico")
# plt.legend()

# plt.plot(
#     filtrados.TZg, filtrados.DHI, 'o', 
#     color="black",markersize=0.1, label="GHI Filtrada")
# #plt.plot(filtrados.TZg, 1.5 * E * filtrados.CTZ**1.2 + 100, 'o', color="red", markersize=0.001, label="Fisc. Posible")
# #plt.plot(filtrados.TZg, 1.2 * E * filtrados.CTZ**1.2 + 50, 'o', color="blue", markersize=0.001, label="Ext. Raros")
# plt.plot(filtrados.TZg, c2 * E * filtrados.CTZ**1.2 + 30, 'o', color="blue", markersize=0.03, label="Lim. Climatológico")
# plt.legend()


# plt.title("DNI Abra Pampa - F Climatico ")
# plt.plot(filtrados.TZg, filtrados.DNI, 'o', markersize=0.1, label="DNI", color="black")
# #plt.plot(filtrados.TZg, E , 'o', color="red", markersize=0.1, label="Fisc. Posible")
# plt.plot(filtrados.TZg, c3 * E * filtrados.CTZ**0.2 + 10, 'o', color="blue", markersize=0.1, label="Lim. Climatológico")
# plt.legend(markerscale=60)
# plt.xlabel("TZ°")
# plt.ylabel("DNI wm²")
# plt.show()


# ctz = filtrados.CTZ
# Prs = 661.12
# Rl = 209.3*ctz- 708.3*ctz**2 + 1128.7*ctz**3 - 911.2*ctz**4 + 287.85*ctz**5 + 0.046725*ctz*Prs


# plt.plot(filtrados.TZg, filtrados.DHI, 'o', color="black", markersize=0.5)
# plt.plot(filtrados.TZg, Rl-1, 'o', color="red", markersize=0.5)




