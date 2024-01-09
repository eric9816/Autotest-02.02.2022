import pandas as pd
from unifloc.common.ambient_temperature_distribution import AmbientTemperatureDistribution
import autotest_core

# -------------------------------------------------------------------------------------------
# Для Pipe

pars_limits = {'pfl': {'inner': [5, 370],
                       'outer': [1, 370]},
               'd': {'inner': [62, 153],
                     'outer': [26.4, 153]},
               'q_fluid': {'inner': [1/86400, 500/86400],
                           'outer': [0.79/86400, 500/86400]},
               'gamma_oil': {'inner': [0.83, 0.9],
                             'outer': [0.7, 0.988]},
               'gamma_gas': {'inner': [0.64, 0.75],
                             'outer': [0.554, 1.4]},
               'gamma_wat': {'inner': [1.01, 1.15],
                             'outer': [1, 1.3]},
               'wct': {'inner': [0, 1],
                       'outer': [0, 1]},
               'GOR': {'inner': [0, 1000],
                       'outer': [0, 15000]},
               'r': {'inner': [0.09, 0.15],
                     'outer': [0, 0.5]},
               'T': {'inner': [293.15, 403.15],
                     'outer': [274.15, 423.15]}}

limited_pars = {'d': 'tubing',
                'roughness': 'tubing'}
# -------------------------------------------------------------------------------------------

fluid_data = {"q_fluid": 9.94994 / 86400,
              "pvt_model_data": {"black_oil": {"gamma_gas": 0.7, "gamma_wat": 1, "gamma_oil": 0.8,
                                "wct": 0.5, "phase_ratio": {"type": "GOR", "value": 500},
                                "oil_correlations":
                                    {"pb": "Standing", "rs": "Standing",
                                     "rho": "Standing", "b": "Standing",
                                     "mu": "Beggs", "compr": "Vasquez"},
                                "gas_correlations": {"ppc": "Standing", "tpc": "Standing",
                                                     "z": "Dranchuk", "mu": "Lee"},
                                "water_correlations": {"b": "McCain", "compr": "Kriel",
                                                       "rho": "Standing", "mu": "McCain"},
                                #"rsb": {"value": 50, "p": 10000000, "t": 303.15},
                                #"muob": {"value": 0.5, "p": 10000000, "t": 303.15},
                                #"bob": {"value": 1.5, "p": 10000000, "t": 303.15},
                                "table_model_data": None, "use_table_model": False,
                                "table_mu": None}}}


pipe_data = {'flowline_length': 100,
             'flowline_angle': -45,
             'flowline_inner_diameter': 40,
             'flowline_roughness': 0.001}

ambient_temperature_data = {'MD': [0, pipe_data['flowline_length']], 'T': [293.15, 298.15]}
amb_temp = AmbientTemperatureDistribution(ambient_temperature_data)

calc_type = 'network'
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
file_path = "D:/scripts/autotest-newb/NEW_autotest/dataset/separate_results/model_test.xlsx"
dataset_path = "D:/scripts/autotest-newb/NEW_autotest/dataset/dataset.csv"
model_path = 'D:/scripts/autotest-newb/NEW_autotest/pips/test_model.pips'

pfl = 100
calc_options = {'error_calc': False,
                'save_results': True,
                'plot_results': True,
                'scenario': False,
                'empty_dataset': False}

if calc_options['empty_dataset']:
    dataset_df = pd.DataFrame(columns=['InclinationAngle',
                                'SuperficialVelocityLiquid',
                                'SuperficialVelocityGas',
                                'DensityLiquidInSitu',
                                'DensityGasInSitu',
                                'SurfaceTensionOilGasInSitu',
                                'PipeInnerDiameter',
                                'PipeRelativeRoughness',
                                'ViscosityLiquidInSitu',
                                'ViscosityGasInSitu',
                                'Pressure',
                                'HoldupFractionLiquid',
                                'FlowPatternGasLiquid',
                                'PressureGradientTotal'])
    dataset_df.to_csv(dataset_path,
                      index=False)

number_of_angles = 20
include0 = False
number_of_samples_per_angle = 50

if not include0:
    step = (90 - 0.5) / (number_of_angles // 2 - 1)
    inclination_angles = [step * i + 0.5 for i in range(6, number_of_angles // 2)]
else:
    inclination_angles = [45, -45]

print(inclination_angles)

if number_of_samples_per_angle > 1:
    sample_flag = True
else:
    sample_flag = False

for inclination_angle in inclination_angles:
    current_flow_direction = 1

    if inclination_angle < 0:
        current_flow_direction = -1

    pipe_data['flowline_angle'] = abs(inclination_angle)

    autotest_core.calc_autotest(p_atma=pfl,
                                file_path=file_path,
                                model_path=model_path,
                                trajectory_data={},
                                ambient_temperature_data=ambient_temperature_data,
                                fluid_data=fluid_data,
                                data=pipe_data,
                                equipment_data=None,
                                calc_type="network",
                                sample=sample_flag,
                                number_of_samples=number_of_samples_per_angle,
                                pars_limits=pars_limits,
                                limited_pars=limited_pars,
                                calc_options=calc_options,
                                hydr_corr_type='gregory',
                                temperature_option='LINEAR',
                                flow_direction=current_flow_direction)