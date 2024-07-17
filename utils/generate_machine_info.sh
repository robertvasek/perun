#!/bin/bash

get_machine_specification() {
  echo -n "{"

  # Get system information
  system=$(uname -s)
  node=$(uname -n)
  release=$(uname -r)
  architecture=$(uname -m)

  echo -n -e "\"system\": \"$system\", "
  echo -n -e "\"host\": \"$node\", "
  echo -n -e "\"release\": \"$release\", "
  echo -n -e "\"architecture\": \"$architecture\","

  # Get CPU information
  physical_cpus=$(lscpu --online --parse=Core,Socket | grep --invert-match '^#' | sort --unique | wc --lines)
  total_cpus=$(nproc)
  cpu_frequency=$(lscpu -p=MHZ | grep -v "^#" | sort -n | tail -1)

  echo -n -e "\"cpu\": {"
  echo -n -e "\"physical\": \"$physical_cpus\","
  echo -n -e "\"total\": \"$total_cpus\","
  echo -n -e "\"frequency\": \"${cpu_frequency}MHz\""
  echo -n -e "}, "

  # Get memory information
  total_ram=$(free -h | grep "^Mem:" | awk '{print $2"B"}')
  swap=$(free -h | grep "^Swap:" | awk '{print $2"B"}')

  echo -n -e "\"memory\": {"
  echo -n -e "\"total_ram\": \"$total_ram\", "
  echo -n -e "\"swap\": \"$swap\""
  echo -n -e "}, "

  # Get boot information from /proc/cmdline
  if [[ -e /proc/cmdline ]]; then
    boot_info=$(cat /proc/cmdline)
    echo -n -e "\"boot_info\": \"$boot_info\","
  fi

  # Get memory details from /proc/meminfo
  echo -n -e "\"mem_details\": {"
  first=0
  if [[ -e /proc/meminfo ]]; then
    while IFS=: read -r key value; do
      if [[ -n "$key" && -n "$value" ]]; then
        value=$(echo -n $value | xargs)
        if [[ $first -eq 0 ]]; then
          first=1
        else
          echo -n -e ", "
        fi
        echo -n -e "\"$key\": \"$value\""
      fi
    done < /proc/meminfo
  fi
  echo -n -e "}, "

  # Get CPU details from /proc/cpuinfo
  echo -n -e "\"cpu_details\": ["
  if [[ -e /proc/cpuinfo ]]; then
    cpu_section=()
    first_proc=0
    while IFS= read -r line; do
      if [[ -z $line ]]; then
        if [[ first_proc -eq 0 ]]; then
          first_proc=1
        else
          echo -n -e ", "
        fi
        echo -n -e "{"
        first_detail=0
        for cpu_detail in "${cpu_section[@]}"; do
          if [[ "$cpu_detail" =~ [^[:space:]] ]]; then
            if [[ first_detail -eq 0 ]]; then
              first_detail=1
            else
              echo -n -e ", "
            fi
            while IFS=: read -r key value; do
              echo -n -e "\"$(echo "$key" | xargs)\": \"$(echo "$value" | xargs)\""
            done <<< "$cpu_detail"
          fi
        done
        cpu_section=()
        echo -n -e "}"
      else
        cpu_section+=("$line")
      fi
    done < /proc/cpuinfo
  fi
  echo -n -e "]"

  echo -n "}"
}

get_machine_specification
