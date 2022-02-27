############################
# Train the Baseline model #
############################

# args=(
#     -name Baseline
#     -module_name baseline -model_name get_baseline  
#     )
# python ../main.py "${args[@]}"

#########################
# Train the U-Net model #
#########################

# args=(
#     -name UNet
#     -module_name unet -model_name get_unet
#     -do_train_feature_enhancement T  
#     )
# python ../main.py "${args[@]}"


##########################
# Train the ExU-Net model#
##########################

args=(
    -name ExUNet
    -module_name exunet -model_name get_exunet  
    -do_train_feature_enhancement T -do_train_embd_enhancement T
    )
python ../main.py "${args[@]}"
