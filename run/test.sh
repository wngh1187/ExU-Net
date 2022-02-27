#######################################
# evaluate the trained Baseline model #
#######################################

# args=(
#     -name Baseline3 -eval T -load_model_path '../weights/Baseline.pt'
#     -module_name baseline -model_name get_baseline  
#     )
# python ../main.py "${args[@]}"


####################################
# evaluate the trained U-Net model #
####################################

# args=(
#     -name UNet -eval T -load_model_path '../weights/U-Net.pt'
#     -module_name unet -model_name get_unet  
#     )
# python ../main.py "${args[@]}"

######################################
# evaluate the trained ExU-Net model #
######################################

args=(
    -name ExUNet -eval T -load_model_path '../weights/ExU-Net.pt'
    -module_name exunet -model_name get_exunet  
    )
python ../main.py "${args[@]}"
