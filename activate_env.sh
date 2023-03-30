
if [[ $1 == 'inf' ]]; then
    source .envs/inference/bin/activate
elif [[ $1 == 'view' ]]; then
    source .envs/view/bin/activate
else
    echo "Invalid env. Must be \"inf\" or \"view\""
    return 0
fi