#!/bin/bash

STUD_ID=${USER#psicourse-stud}

SLEEP_SECS=30

case $STUD_ID in
  [1-9]|10)
    for job in {1..10}; do
      sbatch -J G_T2DAYS --partition=general --time=2-00:00:00 example6_priorities.batch
    done
    ;;
  1[1-9]|20)
    for job in {1..10}; do
      sbatch -J D --partition=daily example6_priorities.batch
    done
    ;;
  2[1-9]|30)
    for job in {1..10}; do
      sbatch -J D_NTASK44 --partition=daily --ntasks=44 example6_priorities.batch
    done
    ;;
  3[1-9]|40)
    for job in {1..10}; do
      if [ $((job % 2)) -eq 0 ]; then
        sbatch -J D_T30n200 --partition=daily --time=00:30:00 --nice=200 example6_priorities.batch
      else
        sbatch -J D_T30 --partition=daily --time=00:30:00 example6_priorities.batch
      fi
    done
    ;;
  4[1-5])
    for job in {1..10}; do
      sbatch -J H --partition=hourly example6_priorities.batch
    done
    ;;
  4[6-9]|50)
    for job in {1..10}; do
      sbatch --partition=hourly example6_priorities.batch
      if [ $((job % 2)) -eq 0 ]; then
        sbatch -J G_T30n200 --partition=general --time=00:30:00 --nice=200 example6_priorities.batch
      else
        sbatch -J G_TASKS44N2 --partition=general --ntasks=44 -N 2 example6_priorities.batch
      fi
    done
    ;;
esac
