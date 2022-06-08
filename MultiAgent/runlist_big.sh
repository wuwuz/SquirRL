#!/bin/bash
# script to generate a list of submit files and submit them to condor
EXEC=$1
runlist=$2
jobname=$3



# set up results directory
dir=$PWD/$jobname/runlist_`date '+%y%m%d_%H.%M.%S'`
echo "Setting up results directory: $dir"
mkdir $PWD/$jobname
mkdir $dir
# preamble
echo "
Executable = $EXEC
Requirements = TARGET.has_avx
Should_Transfer_Files = IF_NEEDED
When_To_Transfer_Output = ON_EXIT
Image_size = 20000000
Rank = Cpus
Request_cpus = 33
Request_gpus = 0
Notification = ERROR
InitialDir = $dir" > $dir/runlist.sub

while read p; do
  echo "$EXEC $p"
  echo "
  Arguments = $p
  Output = process_\$(cluster).\$(process).txt
  Error = process_\$(cluster).\$(process).err
  Log = process_\$(cluster).\$(process).log
  queue" >> $dir/runlist.sub
done <$runlist
#submit to condor
condor_submit $dir/runlist.sub

