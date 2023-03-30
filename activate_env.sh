
if [[ $1 == 'inf' ]]; then
    deactivate
    source .envs/inference/bin/activate
elif [[ $1 == 'view' ]]; then
    deactivate
    source .envs/view/bin/activate
else
    echo "Invalid env"
    exit 1
fi