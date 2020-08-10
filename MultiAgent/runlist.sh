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
Requirements = TARGET.has_avx && ((Machine == \"mittlefrueh.andrew.cmu.edu\") ||(Machine == \"magnum.andrew.cmu.edu\") ||(Machine == \"liberty.andrew.cmu.edu\") ||(Machine == \"horizon.andrew.cmu.edu\") ||(Machine == \"hallertau.andrew.cmu.edu\") ||(Machine == \"golding.andrew.cmu.edu\") ||(Machine == \"galena.andrew.cmu.edu\") ||(Machine == \"fuggle.andrew.cmu.edu\") ||(Machine == \"crystal.andrew.cmu.edu\") ||(Machine == \"columbus.andrew.cmu.edu\") ||(Machine == \"chinook.andrew.cmu.edu\") ||(Machine == \"centennial.andrew.cmu.edu\") ||(Machine == \"hypnos.andrew.local.cmu.edu\") || (Machine == \"hestia.andrew.local.cmu.edu\") ||(Machine == \"hermes.andrew.local.cmu.edu\") ||(Machine == \"helios.andrew.local.cmu.edu\") ||(Machine == \"hebe.andrew.local.cmu.edu\") ||(Machine == \"hades.andrew.local.cmu.edu\") ||(Machine == \"graces.andrew.local.cmu.edu\") ||(Machine == \"gorgons.andrew.local.cmu.edu\") ||(Machine == \"fates.andrew.local.cmu.edu\") ||(Machine == \"fates.andrew.local.cmu.edu\") ||(Machine == \"eris.andrew.local.cmu.edu\") ||(Machine == \"erida.andrew.local.cmu.edu\") ||(Machine == \"cronus.andrew.local.cmu.edu\") ||(Machine == \"demeter.andrew.local.cmu.edu\") ||(Machine == \"ares.andrew.local.cmu.edu\") || (Machine == \"artemis.andrew.local.cmu.edu\"))
Requirements = TARGET.has_avx && ((Machine == \"mittlefrueh.andrew.cmu.edu\") ||(Machine == \"magnum.andrew.cmu.edu\") ||(Machine == \"liberty.andrew.cmu.edu\") ||(Machine == \"horizon.andrew.cmu.edu\") ||(Machine == \"hallertau.andrew.cmu.edu\") ||(Machine == \"golding.andrew.cmu.edu\") ||(Machine == \"galena.andrew.cmu.edu\") ||(Machine == \"fuggle.andrew.cmu.edu\") ||(Machine == \"crystal.andrew.cmu.edu\") ||(Machine == \"columbus.andrew.cmu.edu\") ||(Machine == \"chinook.andrew.cmu.edu\") ||(Machine == \"centennial.andrew.cmu.edu\") ||(Machine == \"hypnos.andrew.local.cmu.edu\")  ||(Machine == \"hermes.andrew.local.cmu.edu\") ||(Machine == \"helios.andrew.local.cmu.edu\") ||(Machine == \"hebe.andrew.local.cmu.edu\") ||(Machine == \"hades.andrew.local.cmu.edu\") ||(Machine == \"graces.andrew.local.cmu.edu\") ||(Machine == \"gorgons.andrew.local.cmu.edu\") ||(Machine == \"fates.andrew.local.cmu.edu\") ||(Machine == \"fates.andrew.local.cmu.edu\") ||(Machine == \"eris.andrew.local.cmu.edu\") ||(Machine == \"erida.andrew.local.cmu.edu\") ||(Machine == \"cronus.andrew.local.cmu.edu\") ||(Machine == \"demeter.andrew.local.cmu.edu\") || (Machine == \"artemis.andrew.local.cmu.edu\"))
Requirements = TARGET.has_avx && ((Machine == \"hypnos.andrew.local.cmu.edu\")  ||(Machine == \"hermes.andrew.local.cmu.edu\") ||(Machine == \"helios.andrew.local.cmu.edu\") ||(Machine == \"hebe.andrew.local.cmu.edu\") ||(Machine == \"hades.andrew.local.cmu.edu\") ||(Machine == \"graces.andrew.local.cmu.edu\") ||(Machine == \"gorgons.andrew.local.cmu.edu\") ||(Machine == \"fates.andrew.local.cmu.edu\") ||(Machine == \"fates.andrew.local.cmu.edu\") ||(Machine == \"eris.andrew.local.cmu.edu\") ||(Machine == \"erida.andrew.local.cmu.edu\") ||(Machine == \"cronus.andrew.local.cmu.edu\") ||(Machine == \"demeter.andrew.local.cmu.edu\") || (Machine == \"artemis.andrew.local.cmu.edu\"))
Requirements = TARGET.has_avx && ((Machine == \"mittlefrueh.andrew.cmu.edu\") ||(Machine == \"magnum.andrew.cmu.edu\") ||(Machine == \"liberty.andrew.cmu.edu\") ||(Machine == \"horizon.andrew.cmu.edu\") ||(Machine == \"hallertau.andrew.cmu.edu\") ||(Machine == \"golding.andrew.cmu.edu\") ||(Machine == \"galena.andrew.cmu.edu\") ||(Machine == \"fuggle.andrew.cmu.edu\") ||(Machine == \"crystal.andrew.cmu.edu\") ||(Machine == \"columbus.andrew.cmu.edu\") ||(Machine == \"chinook.andrew.cmu.edu\") ||(Machine == \"centennial.andrew.cmu.edu\") ||(Machine == \"hypnos.andrew.local.cmu.edu\") || (Machine == \"hestia.andrew.local.cmu.edu\") ||(Machine == \"hermes.andrew.local.cmu.edu\") ||(Machine == \"helios.andrew.local.cmu.edu\") ||(Machine == \"hebe.andrew.local.cmu.edu\") ||(Machine == \"hades.andrew.local.cmu.edu\") ||(Machine == \"graces.andrew.local.cmu.edu\") ||(Machine == \"gorgons.andrew.local.cmu.edu\") ||(Machine == \"fates.andrew.local.cmu.edu\") ||(Machine == \"fates.andrew.local.cmu.edu\") ||(Machine == \"eris.andrew.local.cmu.edu\") ||(Machine == \"erida.andrew.local.cmu.edu\") ||(Machine == \"cronus.andrew.local.cmu.edu\") ||(Machine == \"demeter.andrew.local.cmu.edu\") ||(Machine == \"ares.andrew.local.cmu.edu\") || (Machine == \"artemis.andrew.local.cmu.edu\"))
Requirements = TARGET.has_avx && ((Machine == \"mittlefrueh.andrew.cmu.edu\") ||(Machine == \"magnum.andrew.cmu.edu\") ||(Machine == \"horizon.andrew.cmu.edu\") ||(Machine == \"hallertau.andrew.cmu.edu\") ||(Machine == \"golding.andrew.cmu.edu\") ||(Machine == \"galena.andrew.cmu.edu\") ||(Machine == \"fuggle.andrew.cmu.edu\") ||(Machine == \"crystal.andrew.cmu.edu\") ||(Machine == \"columbus.andrew.cmu.edu\") ||(Machine == \"chinook.andrew.cmu.edu\") ||(Machine == \"centennial.andrew.cmu.edu\") ||(Machine == \"hypnos.andrew.local.cmu.edu\") ||(Machine == \"hermes.andrew.local.cmu.edu\") ||(Machine == \"helios.andrew.local.cmu.edu\") ||(Machine == \"hebe.andrew.local.cmu.edu\") ||(Machine == \"hades.andrew.local.cmu.edu\") ||(Machine == \"graces.andrew.local.cmu.edu\") ||(Machine == \"gorgons.andrew.local.cmu.edu\") ||(Machine == \"fates.andrew.local.cmu.edu\") ||(Machine == \"fates.andrew.local.cmu.edu\") ||(Machine == \"eris.andrew.local.cmu.edu\") ||(Machine == \"erida.andrew.local.cmu.edu\") ||(Machine == \"cronus.andrew.local.cmu.edu\") ||(Machine == \"demeter.andrew.local.cmu.edu\"))
Requirements = TARGET.has_avx && (CUDACapability == 7.5 || (Machine == \"mittlefrueh.andrew.cmu.edu\") ||(Machine == \"magnum.andrew.cmu.edu\") ||(Machine == \"liberty.andrew.cmu.edu\") ||(Machine == \"horizon.andrew.cmu.edu\") ||(Machine == \"hallertau.andrew.cmu.edu\") ||(Machine == \"golding.andrew.cmu.edu\") ||(Machine == \"galena.andrew.cmu.edu\") ||(Machine == \"fuggle.andrew.cmu.edu\") ||(Machine == \"crystal.andrew.cmu.edu\") ||(Machine == \"columbus.andrew.cmu.edu\") ||(Machine == \"chinook.andrew.cmu.edu\") ||(Machine == \"centennial.andrew.cmu.edu\") ||(Machine == \"hypnos.andrew.local.cmu.edu\") || (Machine == \"hestia.andrew.local.cmu.edu\") ||(Machine == \"hermes.andrew.local.cmu.edu\") ||(Machine == \"helios.andrew.local.cmu.edu\") ||(Machine == \"hebe.andrew.local.cmu.edu\") ||(Machine == \"hades.andrew.local.cmu.edu\") ||(Machine == \"graces.andrew.local.cmu.edu\") ||(Machine == \"gorgons.andrew.local.cmu.edu\") ||(Machine == \"fates.andrew.local.cmu.edu\") ||(Machine == \"fates.andrew.local.cmu.edu\") ||(Machine == \"eris.andrew.local.cmu.edu\") ||(Machine == \"erida.andrew.local.cmu.edu\") ||(Machine == \"cronus.andrew.local.cmu.edu\") ||(Machine == \"demeter.andrew.local.cmu.edu\") || (Machine == \"artemis.andrew.local.cmu.edu\"))
Requirements = TARGET.has_avx && ((Machine != \"ares.andrew.cmu.edu\") && (Machine != \"cyh-c0-gpu-000.its.ece.local.cmu.edu\")&& (Machine != \"cyh-c0-gpu-001.its.ece.local.cmu.edu\")&& (Machine != \"cyh-c0-gpu-002.its.ece.local.cmu.edu\"))
Image_size = 20000000
Request_cpus = 8
Request_gpus = 0
Should_Transfer_Files = IF_NEEDED
When_To_Transfer_Output = ON_EXIT
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

