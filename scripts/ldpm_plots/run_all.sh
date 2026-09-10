for i in {1..7}; do
	python plot_ldpm_subfacet_stress_strain.py ldpm_singletet_case$i/ldpm_singletet_case${i}_subfacet_stress_strain.csv -o ldpm_plot_$i.pdf
done
