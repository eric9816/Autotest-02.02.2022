import pandas as pd
from unifloc.common.ambient_temperature_distribution import AmbientTemperatureDistribution
import autotest_core
from unifloc.tools import units_converter as uc

well_trajectory_data = {'inclinometry': pd.DataFrame(columns=['MD', 'TVD'],
                                                 data=[[float(0), float(0)],
                                                       [float(1800), float(1800)]])}
# -------------------------------------------------------------------------------------------
# Для Pipe
pars_limits = {'wct': [0.0, 0.99],
               'q_liq': [0.00005787, 0.005787],
               'rp': [10, 1000],
               'd': [0.06, 0.1],
               't_res': [280.15, 380.15],
               'gamma_oil': [0.65, 0.85]}

limited_pars = {'d': 'tubing',
                'roughness': 'tubing'}
# -------------------------------------------------------------------------------------------
# Для газлифта (well)
# pars_limits = {'wct': [0.0, 0.99],
#                'q_liq': [uc.convert_rate(10, 'm3/day', 'm3/s'),
#                          uc.convert_rate(500, 'm3/day', 'm3/s')],
#                'rp': [10.0, 1000.0],
#                'p_gas_inj': [5000000, 30000000],
#                'freq_q_ag': [10000 / 86400, 100000 / 86400],
#                'pfl': [10, 50],
#                'h_mes': [500, 1100],
#                'p_valve': [30 * 101325, 80 * 101325]}
#
# limited_pars = {'h_mes': 'valve3', 'p_valve': 'valve3'}
# -------------------------------------------------------------------------------------------
table_model_data = {
    'ro': None,
    'rg': None,
    'muo': None,
    'mug': None,
    'bo': None,
    'pb': None,
    'rs': None,
    'z': None,
    'co': None,
    'muw': None,
    'rw': None,
    'stog': None
}
fluid_data = {"q_fluid": 100 / 86400, "wct": 0,
              "pvt_model_data": {"black_oil": {"gamma_gas": 0.6, "gamma_wat": 1, "gamma_oil": 0.8,
                                               "rp": 50,
                                               "oil_correlations": {"pb": "Standing", "rs": "Standing",
                                                                    "rho": "Standing",
                                                                    "b": "Standing", "mu": "Beggs",
                                                                    "compr": "Vasquez"},
                                               "gas_correlations":
                                                   {"ppc": "Standing", "tpc": "Standing", "z": "Dranchuk",
                                                    "mu": "Lee"},
                                               "water_correlations": {"b": "McCain", "compr": "Kriel",
                                                                      "rho": "Standing",
                                                                      "mu": "McCain"},
                                               "rsb": {"value": 300, "p": 10000000, "t": 303.15},
                                               "muob": {"value": 0.5, "p": 10000000, "t": 303.15},
                                               "bob": {"value": 1.5, "p": 10000000, "t": 303.15},
                                               "table_model_data": table_model_data, "use_table_model": None}}}

ambient_temperature_data = {'MD': [0, 1800], 'T': [284.75, 354.75]}
amb_temp = AmbientTemperatureDistribution(ambient_temperature_data)

calc_type = 'pipe'
# -------------------------------------------------------------------------------------------
# для pipe
equipment_data = {'packer': True}
# -------------------------------------------------------------------------------------------
# для газлифта
# equipment_data = {"gl_system": {
#     "valve1": {"h_mes": 1300, "d": 0.003175, "s_bellow": 0.000199677,
#                "p_valve": uc.convert_pressure(50, "atm", "Pa"),
#                "valve_type": "ЦКсОК"},
#     "valve2": {"h_mes": 1100, "d": 0.00396875, "s_bellow": 0.000195483,
#                "p_valve": uc.convert_pressure(60, "atm", "Pa"),
#                "valve_type": "ЦКсОК"},
#     "valve3": {"h_mes": 800, "d": 0.0047625, "s_bellow": 0.000199032,
#                "p_valve": uc.convert_pressure(40, "atm", "Pa"),
#                "valve_type": "ЦКсОК"}}}

# если считаем газлифт, то можно задать давление и расход газлифтного газа при закачке
# qinj = None  # 100000 / 86400
# pinj = None  # 150 * 101325
# -------------------------------------------------------------------------------------------
# для эцн
# motor_data = {"ID": 1,
#               "manufacturer": "Centrilift",
#               "name": "562Centrilift-KMB-130-2200B",
#               "d_motor_mm": 142.7,
#               "motor_nom_i": 35,
#               "motor_nom_power": 96.98,
#               "motor_nom_voltage": 2200,
#               "motor_nom_eff": 80,
#               "motor_nom_cosf": 0.82,
#               "motor_nom_freq": 60,
#               "load_points": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
#               "amperage_points": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
#               "cosf_points": [0.7, 0.74, 0.77, 0.8, 0.83, 0.85, 0.86, 0.87, 0.88, 0.88],
#               "eff_points": [0.78, 0.83, 0.85, 0.88, 0.87, 0.87, 0.87, 0.87, 0.86, 0.86],
#               "rpm_points": [3568.604, 3551.63, 3534.656, 3517.682, 3500.708, 3483.734, 3466.76, 3449.786,
#                              3432.812, 3415.838]
#               }
# equipment_data = {"esp_system": {"esp": {"esp_data": esp_data,
#                                          "stages": 345,
#                                          "viscosity_correction": True,
#                                          "gas_correction": True,
#                                          "gas_degr_value": 0.9},
#                                  "esp_electric_system": {"motor_data": motor_data,
#                                                          "gassep_nom_power": 500,
#                                                          "protector_nom_power": 500,
#                                                          "transform_eff": 0.97,
#                                                          "cs_eff": 0.97,
#                                                          "cable_specific_resistance": 1.18,
#                                                          "cable_length": 1450,
#                                                          },
#                                  "separator": {"k_gas_sep": 0}},
#                                  "packer": False}
# esp_data = None
# esp_id = 8005
# stage_by_stage = False
# freq = 50.667

# -------------------------------------------------------------------------------------------
pipe_data = {"casing": {"bottom_depth": 1800,
                        "d": 0.146,
                        "roughness": 0.0001,
                        "s_wall": 0.005},
             "tubing": {"bottom_depth": 1800,
                        "d": 0.062,
                        "roughness": 0.0001,
                        'ambient_temperature_distribution': amb_temp,
                        "s_wall": 0.005}}
# -------------------------------------------------------------------------------------------
file_path = "C:/Users/PC/PycharmProjects/pythonProject/hagerdon_results/TESSSSSSSSST PIPE ANGLE.xlsx"
model_path = '504.pips'
pfl = 20
calc_options = {'error_calc': True,
                'save_results': True,
                'plot_results': True}

autotest_core.calc_autotest(p_atma=pfl,
                            file_path=file_path,
                            model_path=model_path,
                            trajectory_data=well_trajectory_data,
                            ambient_temperature_data=ambient_temperature_data,
                            fluid_data=fluid_data,
                            pipe_data=pipe_data,
                            equipment_data=equipment_data,
                            calc_type=calc_type,
                            sample=False,
                            number_of_samples=1,
                            pars_limits=pars_limits,
                            limited_pars=limited_pars,
                            calc_options=calc_options,
                            hydr_corr_type='beggsbrill')





