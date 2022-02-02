import pandas as pd

from NEW_autotest.autotest_pipe_subclass import AutotestPipe
from NEW_autotest.autotest_pvt_subclass import AutotestPVT
from NEW_autotest.autotest_gaslift_subclass import AutotestGasLift
from NEW_autotest.autotest_esp_subclass import AutotestESP


def calc_autotest(p_atma: float,
                  file_path: str,
                  model_path: str,
                  trajectory_data: dict,
                  ambient_temperature_data: dict,
                  fluid_data: dict,
                  pipe_data: dict,
                  equipment_data: dict,
                  calc_type: str,
                  esp_data: pd.DataFrame = None,
                  freq: float = None,
                  stage_by_stage: bool = False,
                  p_ann: float = None,
                  sample: bool = False,
                  number_of_samples: int = 5,
                  pars_limits: dict = None,
                  limited_pars: dict = None,
                  calc_options: dict = None,
                  hydr_corr_type: str = 'BeggsBrill',
                  q_gas_inj: float = None,
                  p_gas_inj: float = None,
                  esp_id: int = None,
                  heat_balance: bool = False,
                  temperature_option: str = 'LINEAR',
                  mean_integral_error: bool = False,
                  ):
    """
    :param esp_data: паспортные данные ЭЦН - pd.Series или dict
    :param freq: частота ЭЦН
    :param stage_by_stage: расчет ЭЦН от ступени к ступени
    :param p_ann: давление в затрубном пространстве
    :param esp_id: id насоса
    :param hydr_corr_type: тип гидравлической корреляции
    :param p_atma: давление, атма
    :param file_path: путь сохранения файла с результатами
    :param model_path: путь сохранения файлов в формате pips
    :param trajectory_data: словарь с таблицей с инклинометрией, dict
    :param ambient_temperature_data: словарь с таблицей распределения температуры
    :param fluid_data: словарь с параметрами флюида, dict
    :param pipe_data: словарь с параметрами труб, dict
    :param equipment_data: словарь с параметрами оборудования, dict
    :param calc_type: тип расчета (pvt, well, pipe, esp)
    :param sample: если True, то расчет произведется на произвольном наборе данных.
    В противном случае рассчитается одиночный - main
    :param number_of_samples: количество набота данных (по умолчанию - 5)
    :param pars_limits: словарь с параметром и его границами, в которых будет генерироваться данные, dict
    :param limited_pars: словарь с однозначными соответствиями параметров для объектов, которые могут повторяться, dict
        Например: {'d': 'tubing'} - тогда диаметр заменится только в tubing
    :param calc_options: словарь с параметрами расчета, dict
    :param q_gas_inj: закачка газлифтного газа, ст. м3/с (задается в случае если тип расчета well)
    :param p_gas_inj: давление закачки газлифтного газа, Па (задается в случае если тип расчета well)
    :param heat_balance:  опция учета теплопотерь, по умолчанию - False
    :param temperature_option: 'CONST' - темпертарура константа по всей скважине,
                               'LINEAR' - линейное распределение темпратуры
                               (по умолчанию - CONST)
    :param flow_direction: -1 - расчет от h_start,
                            1 - расчет к h_start
                            (по умолчанию - 1)
    :param h_start: 'top' или 'bottom' (по умолчанию top)
    :param mean_integral_error: True - среднеинтегральная ошибка,
                                False - относительная ошибка
                                (по умолчанию - False)
    """
    if calc_options is None:
        calc_options = {'error_calc': True,
                        'save_results': None,
                        'plot_results': None}
    if heat_balance:
        raise ValueError('Учет теплопотерь не реализован')

    if sample:  # Если sample = True, значит расчет моделей будет на произвольном наборе параметров
        if pars_limits is None:  # Проверка, есть ли данные по pars_limits
            raise ValueError('Нет данных pars_limits')

        if calc_type.lower() == 'pvt':  # тип расчета - PVT
            pvt = AutotestPVT(model_path,
                              file_path,
                              trajectory_data,
                              ambient_temperature_data,
                              fluid_data,
                              pipe_data,
                              equipment_data,
                              hydr_corr_type)

            pvt.sample_model(pars_limits=pars_limits,
                             limited_pars=limited_pars,
                             number_of_samples=number_of_samples,
                             pfl=p_atma,
                             model_path=model_path,
                             fluid_data=fluid_data,
                             pipe_data=pipe_data,
                             calc_type=calc_type,
                             equipment_data=equipment_data,
                             well_trajectory_data=trajectory_data,
                             calc_options=calc_options,
                             result_path=file_path,
                             heat_balance=heat_balance,
                             temperature_option=temperature_option,
                             flow_direction=1,
                             h_start='top',
                             mean_integral_error=mean_integral_error,
                             ambient_temperature_data=ambient_temperature_data)

        elif calc_type.lower() == 'esp':
            esp = AutotestESP(model_path,
                              file_path,
                              trajectory_data,
                              ambient_temperature_data,
                              fluid_data,
                              pipe_data,
                              equipment_data,
                              hydr_corr_type)

            esp.sample_model(pars_limits=pars_limits,
                             limited_pars=limited_pars,
                             number_of_samples=number_of_samples,
                             pfl=p_atma,
                             model_path=model_path,
                             fluid_data=fluid_data,
                             pipe_data=pipe_data,
                             esp_data=esp_data,
                             esp_id=esp_id,
                             stage_by_stage=stage_by_stage,
                             freq_q_ag=freq,
                             calc_type=calc_type,
                             equipment_data=equipment_data,
                             well_trajectory_data=trajectory_data,
                             calc_options=calc_options,
                             result_path=file_path,
                             heat_balance=heat_balance,
                             temperature_option=temperature_option,
                             flow_direction=1,
                             h_start='top',
                             mean_integral_error=mean_integral_error,
                             ambient_temperature_data=ambient_temperature_data,
                             p_ann=p_ann)

        elif calc_type.lower() == 'pipe':
            pipe = AutotestPipe(model_path,
                                file_path,
                                trajectory_data,
                                ambient_temperature_data,
                                fluid_data,
                                pipe_data,
                                equipment_data,
                                hydr_corr_type)

            pipe.sample_model(pars_limits=pars_limits,
                              limited_pars=limited_pars,
                              number_of_samples=number_of_samples,
                              pfl=p_atma,
                              model_path=model_path,
                              fluid_data=fluid_data,
                              pipe_data=pipe_data,
                              calc_type=calc_type,
                              equipment_data=equipment_data,
                              well_trajectory_data=trajectory_data,
                              calc_options=calc_options,
                              result_path=file_path,
                              heat_balance=heat_balance,
                              temperature_option=temperature_option,
                              flow_direction=1,
                              h_start='top',
                              mean_integral_error=mean_integral_error,
                              ambient_temperature_data=ambient_temperature_data)

        elif calc_type.lower() == 'well':  # в тип расчета well будет включен также и ЭЦН (пока только газлифт)
            if "gl_system" in equipment_data:
                gaslift = AutotestGasLift(model_path,
                                          file_path,
                                          trajectory_data,
                                          ambient_temperature_data,
                                          fluid_data,
                                          pipe_data,
                                          equipment_data,
                                          hydr_corr_type)

                gaslift.sample_model(pars_limits=pars_limits,
                                     limited_pars=limited_pars,
                                     number_of_samples=number_of_samples,
                                     pfl=p_atma,
                                     model_path=model_path,
                                     fluid_data=fluid_data,
                                     pipe_data=pipe_data,
                                     equipment_data=equipment_data,
                                     well_trajectory_data=trajectory_data,
                                     calc_type=calc_type,
                                     result_path=file_path,
                                     freq_q_ag=q_gas_inj,
                                     calc_options=calc_options,
                                     temperature_option=temperature_option,
                                     heat_balance=heat_balance,
                                     ambient_temperature_data=ambient_temperature_data,
                                     p_gas_inj=p_gas_inj)

            elif 'esp_system' in equipment_data:
                if equipment_data['esp_system']['esp_electric_system'] is None:
                    raise ValueError('Для расчета скважины с ЭЦН нужны данные по электрической системе насоса '
                                     '(esp_electric_system)')
                if p_ann is None:
                    raise ValueError('Нужно задать затрубное давление p_ann')

                esp = AutotestESP(model_path,
                                  file_path,
                                  trajectory_data,
                                  ambient_temperature_data,
                                  fluid_data,
                                  pipe_data,
                                  equipment_data,
                                  hydr_corr_type)

                esp.sample_model(pars_limits=pars_limits,
                                 limited_pars=limited_pars,
                                 number_of_samples=number_of_samples,
                                 pfl=p_atma,
                                 model_path=model_path,
                                 fluid_data=fluid_data,
                                 pipe_data=pipe_data,
                                 esp_data=esp_data,
                                 esp_id=esp_id,
                                 stage_by_stage=stage_by_stage,
                                 freq_q_ag=freq,
                                 calc_type=calc_type,
                                 equipment_data=equipment_data,
                                 well_trajectory_data=trajectory_data,
                                 calc_options=calc_options,
                                 result_path=file_path,
                                 heat_balance=heat_balance,
                                 temperature_option=temperature_option,
                                 flow_direction=1,
                                 h_start='top',
                                 mean_integral_error=mean_integral_error,
                                 ambient_temperature_data=ambient_temperature_data,
                                 p_ann=p_ann)

    else:  # если Sample = False, значит посчитается единичный расчет main по аналогичному алгоритму
        if calc_type.lower() == 'pvt':
            pvt = AutotestPVT(model_path,
                              file_path,
                              trajectory_data,
                              ambient_temperature_data,
                              fluid_data,
                              pipe_data,
                              equipment_data,
                              hydr_corr_type)

            pvt.main(pfl=p_atma,
                     model_path=model_path,
                     fluid_data=fluid_data,
                     pipe_data=pipe_data,
                     calc_type=calc_type,
                     equipment_data=equipment_data,
                     well_trajectory_data=trajectory_data,
                     calc_options=calc_options,
                     result_path=file_path,
                     heat_balance=heat_balance,
                     temperature_option=temperature_option,
                     flow_direction=1,
                     h_start='top',
                     mean_integral_error=mean_integral_error,
                     ambient_temperature_data=ambient_temperature_data)

        elif calc_type.lower() == 'pipe':
            pipe = AutotestPipe(model_path,
                                file_path,
                                trajectory_data,
                                ambient_temperature_data,
                                fluid_data,
                                pipe_data,
                                equipment_data,
                                hydr_corr_type)

            pipe.main(pfl=p_atma,
                      model_path=model_path,
                      fluid_data=fluid_data,
                      pipe_data=pipe_data,
                      calc_type=calc_type,
                      equipment_data=equipment_data,
                      well_trajectory_data=trajectory_data,
                      calc_options=calc_options,
                      result_path=file_path,
                      heat_balance=heat_balance,
                      temperature_option=temperature_option,
                      flow_direction=1,
                      h_start='top',
                      mean_integral_error=mean_integral_error,
                      ambient_temperature_data=ambient_temperature_data)

        elif calc_type.lower() == 'esp':
                esp = AutotestESP(model_path,
                                  file_path,
                                  trajectory_data,
                                  ambient_temperature_data,
                                  fluid_data,
                                  pipe_data,
                                  equipment_data,
                                  hydr_corr_type)

                esp.main(pfl=p_atma,
                         model_path=model_path,
                         fluid_data=fluid_data,
                         pipe_data=pipe_data,
                         esp_data=esp_data,
                         esp_id=esp_id,
                         freq=freq,
                         stage_by_stage=stage_by_stage,
                         calc_type=calc_type,
                         equipment_data=equipment_data,
                         well_trajectory_data=trajectory_data,
                         calc_options=calc_options,
                         result_path=file_path,
                         heat_balance=heat_balance,
                         temperature_option=temperature_option,
                         flow_direction=1,
                         h_start='top',
                         mean_integral_error=mean_integral_error,
                         ambient_temperature_data=ambient_temperature_data,
                         p_ann=p_ann)

        elif calc_type.lower() == 'well':
            if "gl_system" in equipment_data:
                gaslift = AutotestGasLift(model_path,
                                          file_path,
                                          trajectory_data,
                                          ambient_temperature_data,
                                          fluid_data,
                                          pipe_data,
                                          equipment_data,
                                          hydr_corr_type)

                gaslift.main(pfl=p_atma,
                             result_path=file_path,
                             model_path=model_path,
                             fluid_data=fluid_data,
                             pipe_data=pipe_data,
                             equipment_data=equipment_data,
                             well_trajectory_data=trajectory_data,
                             calc_type=calc_type,
                             freq=q_gas_inj,
                             calc_options=calc_options,
                             temperature_option=temperature_option,
                             heat_balance=heat_balance,
                             ambient_temperature_data=ambient_temperature_data,
                             p_gas_inj=p_gas_inj)

            elif 'esp_system' in equipment_data:
                if equipment_data['esp_system']['esp_electric_system'] is None:
                    raise ValueError('Для расчета скважины с ЭЦН нужны данные по электрической системе насоса '
                                     '(esp_electric_system)')
                if p_ann is None:
                    raise ValueError('Нужно задать затрубное давление p_ann')

                esp = AutotestESP(model_path,
                                  file_path,
                                  trajectory_data,
                                  ambient_temperature_data,
                                  fluid_data,
                                  pipe_data,
                                  equipment_data,
                                  hydr_corr_type)

                esp.main(pfl=p_atma,
                         model_path=model_path,
                         fluid_data=fluid_data,
                         pipe_data=pipe_data,
                         esp_data=esp_data,
                         esp_id=esp_id,
                         freq=freq,
                         stage_by_stage=stage_by_stage,
                         calc_type=calc_type,
                         equipment_data=equipment_data,
                         well_trajectory_data=trajectory_data,
                         calc_options=calc_options,
                         result_path=file_path,
                         heat_balance=heat_balance,
                         temperature_option=temperature_option,
                         flow_direction=1,
                         h_start='top',
                         mean_integral_error=mean_integral_error,
                         ambient_temperature_data=ambient_temperature_data,
                         p_ann=p_ann)
