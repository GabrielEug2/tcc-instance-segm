# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

VENVS_DIR="${SCRIPT_DIR}/.venvs"

discovered_envs=()
for file in ${VENVS_DIR}/*; do
	if [[ -d $file ]] ; then
		env_name="${file##*/}"
		discovered_envs+=("$env_name")
	fi
done
if [[ "${#discovered_envs[@]}" -eq 0 ]]; then
	echo "No envs found on ${VENVS_DIR}"
fi

select env in "${discovered_envs[@]}"
do
	valid_option=false
	for valid_env in "${discovered_envs[@]}"; do
		if [[ $env == $valid_env ]]; then
			valid_option=true
			break
		fi
	done

	if [[ $valid_option == true ]]; then
		deactivate
		source ${VENVS_DIR}/${env}/bin/activate
	else
		echo "invalid option"
	fi
	break
done

