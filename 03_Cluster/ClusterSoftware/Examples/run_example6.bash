#!/bin/bash

STUD_ID=${USER#psicourse-stud}

SLEEP_SECS=30

case $STUD_ID in
  [1-9]|10)
    for job in {1..10}; do
      sbatch --partition=general example6_priorities.batch
    done
    ;;
  1[1-9]|20)
    for job in {1..10}; do
      sbatch --partition=daily example6_priorities.batch
    done
    ;;
  2[1-9]|30)
    for job in {1..10}; do
      sbatch --partition=daily --ntasks=44 example6_priorities.batch
    done
    ;;
  3[1-9]|40)
    for job in {1..10}; do
      if [ $((job % 2)) -eq 0 ]; then
        sbatch --partition=daily --time=00:30:00 --nice=200 example6_priorities.batch
      else
        sbatch --partition=daily --time=00:30:00 example6_priorities.batch
      fi
    done
    ;;
  4[1-5])
    for job in {1..10}; do
      sbatch --partition=hourly example6_priorities.batch
    done
    ;;
  4[6-9]|50)
    for job in {1..10}; do
      sbatch --partition=hourly example6_priorities.batch
      if [ $((job % 2)) -eq 0 ]; then
        sbatch --partition=daily --time=00:30:00 --nice=200 example6_priorities.batch
      else
        sbatch --partition=daily --ntasks=44 -N 2 example6_priorities.batch
      fi
    done
    ;;
esac
