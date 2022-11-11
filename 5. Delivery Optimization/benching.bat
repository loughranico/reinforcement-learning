@echo off
title Hehe
echo Willkommen 
python main_benchmarking.py 100000 tiny2 tiempo_fijo
python main_benchmarking.py 50000 small2 tiempo_fijo
python main_benchmarking.py 6000 medium2 tiempo_fijo
python main_benchmarking.py 2250 large2 tiempo_fijo
python main_benchmarking.py 250000 tiny2 mejor_coste
python main_benchmarking.py 500000 small2 mejor_coste
python main_benchmarking.py 500000 medium2 mejor_coste
python main_benchmarking.py 250000 large2 mejor_coste