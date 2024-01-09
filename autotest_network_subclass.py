"""
Подкласс для автотестирования трубы (pipe).

Позволяет как вызывать сравнительный расчет вручную, так и на рандомных данных в определенных диапазонах данных,
генерируемых с помощью латинского гиперкуба.

31/01/2022

@alexey_vodopyan
@erik_ovsepyan
"""
import math
import random

import pandas as pd
from smt.sampling_methods import LHS
from sixgill.pipesim import Model
from typing import Tuple, Union, Any
from sixgill.definitions import ModelComponents, Constants, Units, Parameters, ProfileVariables, SystemVariables
import numpy as np
from copy import deepcopy
from unifloc.tools import units_converter as uc
from autotest_class import Autotest


def generate_custom_sample(pars_limits,
                           number_of_samples,
                           out_of_bounds_frac=0.05):
    def ifm(x,
            limits,
            out_of_bounds_frac):
        left_gap = limits['inner'][0] - limits['outer'][0]
        right_gap = limits['outer'][1] - limits['inner'][1]

        if left_gap == 0 and right_gap == 0:
            return limits['inner'][0] + x * (limits['inner'][1] - limits['inner'][0])
        elif left_gap == 0 and right_gap != 0:
            if x <= 1 - out_of_bounds_frac:
                return (limits['inner'][1] - limits['inner'][0]) / (1 - out_of_bounds_frac) + limits['inner'][0]
            else:
                return (x + out_of_bounds_frac - 1) * right_gap / out_of_bounds_frac + limits['inner'][1]
        elif left_gap != 0 and right_gap == 0:
            if x <= out_of_bounds_frac:
                return left_gap / out_of_bounds_frac * x + limits['outer'][0]
            else:
                return (limits['inner'][1] - limits['inner'][0]) / (1 - out_of_bounds_frac) * (x - out_of_bounds_frac) + \
                    limits['inner'][0]
        else:
            c1 = out_of_bounds_frac * left_gap / (left_gap + right_gap)
            c2 = out_of_bounds_frac * right_gap / (left_gap + right_gap)

            if x <= c1:
                return left_gap / c1 * x + limits['outer'][0]
            elif x <= 1 - out_of_bounds_frac + c1:
                return (limits['inner'][1] - limits['inner'][0]) / (1 - out_of_bounds_frac) * (x - c1) + \
                    limits['inner'][0]
            else:
                return right_gap / c2 * (x - c1 + out_of_bounds_frac - 1) + limits['inner'][1]

    parameters_df = pd.DataFrame(columns=pars_limits.keys())
    for sample_index in range(number_of_samples):
        for key in pars_limits.keys():
            if key == 'q_fluid':
                pars_limits['q_fluid'] = {'inner': [10 / 86400, 1000 / 86400],
                                          'outer': [1 / 86400, 50 * math.pi / 4 * parameters_df.at[sample_index,\
                                              'd'] ** 2]}
            parameters_df.at[sample_index, key] = ifm(random.random(),
                                                      limits=pars_limits[key],
                                                      out_of_bounds_frac=out_of_bounds_frac)
    return parameters_df


class AutotestNetwork(Autotest):

    def __init__(self,
                 model_path,
                 file_path,
                 trajectory_data,
                 ambient_temperature_data,
                 fluid_data,
                 pipe_data,
                 equipment_data,
                 hydr_corr_type):

        super().__init__(model_path,
                         file_path,
                         trajectory_data,
                         ambient_temperature_data,
                         fluid_data,
                         pipe_data,
                         equipment_data,
                         hydr_corr_type)

        self.L_REPORT = 1

    def find_change_parameter(self,
                              keys: list,
                              data: list,
                              pfl: float,
                              model_path: str,
                              fluid_data: dict,
                              pipe_data: dict,
                              limited_pars: dict,
                              equipment_data: dict = None,
                              freq_q_ag: float = None,
                              **kwargs) -> \
            Tuple[Union[float, Any], str, dict, dict, dict, str]:
        """
        Функция для обновления значений переменных соответственно значениям латинского гиперкуба

        Parameters
        ----------
        :param keys: названия параметров для которых проводим семплирование, list
        :param data: строка с одним набором параметров, list
        :param pfl: линейное давление, атма, float
        :param model_path: путь к файлу с моделью, string
        :param fluid_data: словарь с параметрами флюида, dict
        :param pipe_data: словарь с параметрами труб, dict
        :param limited_pars: словарь с однозначными соответствиями параметров для объектов, которые могут повторяться, dict
        Например: {'d': 'tubing'} - тогда диаметр заменится только в tubing

        Returns
        -------

        """
        # Разделим model_path
        model_path_new = model_path[:model_path.find('.pips')] + '_ang_' + str(pipe_data['flowline_angle'])

        pfl_new = None
        fluid_data_new = {'pvt_model_data': {'black_oil': {'phase_ratio': {'type': 'GOR',
                                                                           'value': None}}}}
        ambient_temperature_data_new = {'MD': [0, pipe_data['flowline_length']], 'T': [None, None]}
        pipe_data_new = pipe_data
        init_dicts = [fluid_data, pipe_data]

        name_postfix = '_ang_' + str(round(pipe_data['flowline_angle'], 3))

        # Цикл поиска ключей в исходных данных
        for i in range(len(keys)):
            if keys[i] == 'pfl':
                pfl_new = data[i]
                model_path_new += '_%s_%s' % (keys[i], str(round(data[i], 3)))
                name_postfix += '_%s_%s' % (keys[i], str(round(data[i], 3)))
            elif keys[i] == 'GOR':
                fluid_data_new['pvt_model_data']['black_oil']['phase_ratio']['value'] = data[i]
                model_path_new += '_%s_%s' % (keys[i], str(round(data[i], 3)))
                name_postfix += '_%s_%s' % (keys[i], str(round(data[i], 3)))
            elif keys[i] == 'T':
                ambient_temperature_data_new['T'] = [data[i], data[i] + 5]
                model_path_new += '_%s_%s' % (keys[i], str(round(data[i], 3)))
                name_postfix += '_%s_%s' % (keys[i], str(round(data[i], 3)))
            elif keys[i] == 'd':
                pipe_data_new['flowline_inner_diameter'] = data[i]
                model_path_new += '_%s_%s' % (keys[i], str(round(data[i], 3)))
                name_postfix += '_%s_%s' % (keys[i], str(round(data[i], 3)))
            elif keys[i] == 'r':
                pipe_data_new['flowline_roughness'] = data[i]
                model_path_new += '_%s_%s' % (keys[i], str(round(data[i], 3)))
                name_postfix += '_%s_%s' % (keys[i], str(round(data[i], 3)))
            else:
                for j in range(len(init_dicts)):

                    # Запустим рекурсивную процедуру поиска ключа по словарю, включая вложенные словари
                    dic_new, flag_find = self.find_key_in_dict(keys[i], data[i], init_dicts[j], limited_pars)

                    # Если ключ найден
                    if flag_find:
                        if j == 0:
                            fluid_data_new = dic_new
                        elif j == 1:
                            pipe_data_new = dic_new

                        model_path_new += '_%s_%s' % (keys[i], str(round(data[i], 3)))
                        name_postfix += '_%s_%s' % (keys[i], str(round(data[i], 3)))

                        # Заменим элемент, тк следующий ключ может попасться в этом же словаре
                        init_dicts[j] = dic_new

                        # Нет смысла продолжать поиск, тк каждый ключ уникален
                        break

        # Вернем путь в формат Pipesim
        model_path_new += '.pips'

        if pfl_new is None:
            pfl_new = pfl

        if fluid_data_new is None:
            fluid_data_new = fluid_data

        if pipe_data_new is None:
            pipe_data_new = pipe_data

        if ambient_temperature_data_new is None:
            ambient_temperature_data_new = kwargs['ambient_temperature_data']

        return pfl_new, model_path_new, fluid_data_new, pipe_data_new, ambient_temperature_data_new, name_postfix

    def reformat_results(self,
                         df,
                         calc_type=None):
        df.rename(columns={'pfl': 'pfl, atma',
                           'q_fluid': 'q_fluid, m3/day',
                           'gamma_oil': 'gamma_oil, sg',
                           'gamma_gas': 'gamma_gas, sg',
                           'gamma_wat': 'gamma_wat, sg',
                           'wct': 'wct, %',
                           'GOR': 'GOR, m3/m3',
                           'd': 'd (inner diameter), mm',
                           'r': 'r (roughness), mm',
                           'T': 'T, K'
                           },
                  inplace=True)
        return df

    def sample_model(self,
                     pars_limits: dict,
                     number_of_samples: int,
                     pfl: float,
                     model_path: str,
                     fluid_data: dict,
                     pipe_data: dict,
                     calc_options: dict,
                     temperature_option: str = None,
                     limited_pars: dict = None,
                     equipment_data: dict = None,
                     calc_type: str = 'well',
                     result_path: str = 'results.xlsx',
                     heat_balance: bool = False,
                     flow_direction=None,
                     h_start=None,
                     ambient_temperature_data=None):
        """
        Функция для расчета моделей на произвольном наборе параметров

        Parameters
        ----------
        :param h_start:
        :param heat_balance:
        :param flow_direction:
        :param ambient_temperature_data:
        :param pars_limits: словарь с параметром и его границами, в которых будет генерироваться данные, dict
        :param limited_pars: словарь с однозначными соответствиями параметров для объектов, которые могут повторяться, dict
            Например: {'d': 'tubing'} - тогда диаметр заменится только в tubing
        :param number_of_samples: количество наборов параметров, integer
        :param pfl: линейное давление, атма, float
        :param model_path: путь к файлу с моделью, string
        :param fluid_data: словарь с параметрами флюида, dict
        :param pipe_data: словарь с параметрами труб, dict
        :param calc_options: словарь с параметрами расчета, dict
        :param temperature_option: опция для расчета температуры, 'Const' или 'Linear', string
        :param equipment_data: словарь с параметрами оборудования, dict
        :param calc_type: цель расчета (Что мы хотим посчитать и сравнить? 'well', 'pvt', 'esp', 'pipe'), str
        :param result_path: путь к файлу с результатами, str
        Returns
        -------

        """

        # Подготовка границ для генерации латинского гиперкуба
        keys = list(pars_limits.keys())

        data = generate_custom_sample(pars_limits=pars_limits,
                                      number_of_samples=number_of_samples,
                                      out_of_bounds_frac=0.05)
        # Результирующая таблица
        results_df = pd.DataFrame(columns=keys + ['Density_inversion_flag'],
                                  index=range(number_of_samples))

        # Итерируемся по набору данных и считаем модели
        for i in range(number_of_samples):
            print('Текущий угол: {}. Расчет {} из {}...'.format(pipe_data['flowline_angle'] * flow_direction,
                                                                i + 1,
                                                                number_of_samples))

            # Определим, что за параметры и изменим их значения
            pfl_new, model_path_new, fluid_data_new, pipe_data_new, new_ambient_temperature_data, name_postfix = \
                self.find_change_parameter(keys=keys,
                                           data=data.iloc[i].to_list(),
                                           pfl=pfl,
                                           model_path=model_path,
                                           fluid_data=fluid_data,
                                           pipe_data=pipe_data,
                                           equipment_data=equipment_data,
                                           limited_pars=limited_pars,
                                           ambient_temperature_data=ambient_temperature_data)
            result_path_new = result_path[:result_path.find('.xlsx')] + name_postfix + '.xlsx'

            # Передадим в pipesim и проведем расчет
            try:
                results_dict = self.main(fluid_data=fluid_data_new,
                                         pipe_data=pipe_data_new,
                                         model_path=model_path_new,
                                         temperature_option=temperature_option,
                                         pfl=pfl_new,
                                         heat_balance=heat_balance,
                                         ambient_temperature_data=new_ambient_temperature_data,
                                         flow_direction=flow_direction,
                                         h_start=h_start,
                                         calc_options=calc_options,
                                         result_path=result_path_new)
            except Exception:
                continue

            # Сохраним значения аргументов
            results_df.loc[i, keys] = data.iloc[i].to_list()
            results_df.loc[i, 'Density_inversion_flag'] = results_dict['density_inversion']

        # Сконвертируем дебит в м3/сут
        if 'q_fluid' in results_df.columns:
            results_df['q_fluid'] = uc.convert_rate(results_df['q_fluid'], 'm3/s', 'm3/day')

        # Приведем результаты к удобному для вывода формату
        results_df.dropna(how='all', inplace=True)
        if not calc_options['scenario']:
            if len(results_df) > 0:
                results_df = self.reformat_results(results_df, calc_type)
                results_df.to_excel(result_path)

        return results_df

    def calc_model_pipesim(self,
                           pfl: float,
                           model_path: str,
                           fluid_data: dict,
                           pipe_data: dict,
                           heat_balance=None,
                           temperature_option: str = 'CONST',
                           flow_direction=None,
                           h_start=None,
                           ambient_temperature_data=None
                           ):

        fluid_data_new = deepcopy(fluid_data)

        # Параметры для создания модели
        well_name = "FL-1"
        qliq = fluid_data_new['q_fluid'] * 86400

        pvt_model_data = fluid_data_new['pvt_model_data']
        black_oil_model = pvt_model_data['black_oil']

        wct = black_oil_model['wct'] * 100
        gamma_oil = black_oil_model['gamma_oil']
        dod = gamma_oil * 1000
        gamma_water = black_oil_model['gamma_wat']
        gamma_gas = black_oil_model['gamma_gas']
        gor = black_oil_model['phase_ratio']['value']
        t_res = ambient_temperature_data["T"][1] - 273.15

        flowline_data = {'flowline_length': pipe_data['flowline_length'],
                         'flowline_angle': pipe_data['flowline_angle'],
                         'flowline_inner_diameter': pipe_data['flowline_inner_diameter'],
                         'flowline_roughness': pipe_data['flowline_roughness']}

        # Создадим модель, сохраним и закроем
        print('Model path: {}'.format(model_path))
        model = Model.new(model_path, units=Units.METRIC, overwrite=True)
        model.save()
        model.close()

        # Откроем модель снова, чтобы были метрические единицы измерения
        model = Model.open(model_path, units=Units.METRIC)

        self.pipesim_model(model=model,
                           modelcomponents=ModelComponents,
                           temperature_option=temperature_option,
                           flowline_data=flowline_data,
                           flow_direction=flow_direction,
                           h_start=h_start,
                           gor=gor,
                           wct=wct,
                           dod=dod,
                           gamma_water=gamma_water,
                           gamma_gas=gamma_gas,
                           well_name=well_name,
                           t1=ambient_temperature_data['T'][0],
                           pfl=pfl)

        parameters = self.pipesim_parameters(wct=wct,
                                             model=model,
                                             black_oil_model=black_oil_model,
                                             heat_balance=heat_balance,
                                             temperature_option=temperature_option,
                                             ambient_temperature_data=ambient_temperature_data,
                                             constants=Constants,
                                             hydr_corr_type=self.hydr_corr_type,
                                             t_res=t_res,
                                             p=pfl,
                                             well_name=well_name,
                                             qliq=qliq,
                                             h_cas=flowline_data['flowline_length'])

        system_variables = [
            SystemVariables.PRESSURE,
            SystemVariables.TEMPERATURE]

        results = model.tasks.ptprofilesimulation.run(producer='Src-1',
                                                      parameters=parameters[0],
                                                      profile_variables=self.profile_variables,
                                                      system_variables=system_variables)
        model.save()
        model.close()

        global PRESSURE_INDEX
        # self.PRESSURE_INDEX = uc.convert_pressure(results.profile[results.cases[0]]['Pressure'], 'bar', 'mpa')[3:]

        global TEMPERATURE_INDEX
        # self.TEMPERATURE_INDEX = uc.convert_temperature(results.profile[results.cases[0]]['Temperature'], 'C', 'K')[3:]

        global DEPTH
        # self.DEPTH = results.profile[results.cases[0]][ProfileVariables.MEASURED_DEPTH][:1:-1]

        # flag = self.analyze_density_inversion(results)
        flag = False

        return results, fluid_data_new, flag

    def pipesim_parameters(self,
                           wct,
                           model,
                           well_name,
                           black_oil_model,
                           heat_balance,
                           temperature_option,
                           ambient_temperature_data,
                           constants,
                           p,
                           qliq,
                           hydr_corr_type,
                           t_res,
                           h_cas=None,
                           equipment_data=None,
                           h_tub=None,
                           modelcomponents=None):
        # Случай чистой воды
        if wct == 100:
            updated_params = {"BK111": {Parameters.BlackOilFluid.USEGASRATIO: 'GLR',
                                        Parameters.BlackOilFluid.GLR: 0}}
            model.set_values(updated_params)

        # Проверка есть ли калибровочные значения параметров pb, rsb, bob, muob, задание если есть
        if 'rsb' in black_oil_model:
            if black_oil_model['rsb'] is not None:
                updated_params = {"BK111": {Parameters.BlackOilFluid.SinglePointCalibration.
                                            BUBBLEPOINTSATGAS_VALUE: black_oil_model['rsb']['value'],
                                            Parameters.BlackOilFluid.SinglePointCalibration.
                                            BUBBLEPOINTSATGAS_PRESSURE:
                                                uc.convert_pressure(black_oil_model['rsb']['p'], 'pa', 'bar'),
                                            Parameters.BlackOilFluid.SinglePointCalibration.
                                            BUBBLEPOINTSATGAS_TEMPERATURE:
                                                uc.convert_temperature(black_oil_model['rsb']['t'], 'k', 'c')}}
                model.set_values(updated_params)

        if 'bob' in black_oil_model:
            if black_oil_model['bob'] is not None:
                updated_params = {"BK111": {Parameters.BlackOilFluid.SinglePointCalibration.
                                            BELOWBBPOFVF_VALUE: black_oil_model['bob']['value'],
                                            Parameters.BlackOilFluid.SinglePointCalibration.
                                            BELOWBBPOFVF_PRESSURE:
                                                uc.convert_pressure(black_oil_model['bob']['p'], 'pa', 'bar'),
                                            Parameters.BlackOilFluid.SinglePointCalibration.BELOWBBPOFVF_TEMPERATURE:
                                                uc.convert_temperature(black_oil_model['bob']['t'], 'k', 'c')}}
                model.set_values(updated_params)

        if 'muob' in black_oil_model:
            if black_oil_model['muob'] is not None:
                updated_params = {"BK111": {Parameters.BlackOilFluid.SinglePointCalibration.
                                            BELOWBBPLIVEOILVISCOSITY_VALUE: black_oil_model['muob']['value'],
                                            Parameters.BlackOilFluid.SinglePointCalibration.
                                            BELOWBBPLIVEOILVISCOSITY_TEMPERATURE:
                                                uc.convert_temperature(black_oil_model['muob']['t'], 'k', 'c'),
                                            Parameters.BlackOilFluid.SinglePointCalibration.
                                            BELOWBBPLIVEOILVISCOSITY_PRESSURE:
                                                uc.convert_pressure(black_oil_model['muob']['p'], 'pa', 'bar')}}
                model.set_values(updated_params)

        # Установим температуру
        if heat_balance:
            heat_balance_pipesim = 'HEAT BALANCE = ON'
        else:
            heat_balance_pipesim = 'HEAT BALANCE = OFF'
        if temperature_option == 'CONST':
            model.sim_settings[Parameters.SimulationSetting.SINGLEBRANCHKEYWORDS] = heat_balance_pipesim
            # geothermal_survey = {Parameters.GeothermalSurvey.MEASUREDDISTANCE: [0.0, h_cas],
            #                      Parameters.GeothermalSurvey.TEMPERATURE: [float(t_res), float(t_res)]}
            # geothermal_df = pd.DataFrame(geothermal_survey)
            # model.set_geothermal_profile(Well=well_name, value=geothermal_df)
            model.sim_settings[Parameters.SimulationSetting.AMBIENTTEMPERATURE] = ambient_temperature_data['T'][
                                                                                      0] - 293.15
        elif temperature_option == 'LINEAR':
            model.sim_settings[Parameters.SimulationSetting.SINGLEBRANCHKEYWORDS] = heat_balance_pipesim
            # geothermal_survey = {Parameters.GeothermalSurvey.MEASUREDDISTANCE: [0.0, h_cas],
            #                      Parameters.GeothermalSurvey.TEMPERATURE: [
            #                          float(ambient_temperature_data['T'][0]) - 273.15,
            #                          float(t_res)]}
            #
            # geothermal_df = pd.DataFrame(geothermal_survey)
            # model.set_geothermal_profile(Well=well_name, value=geothermal_df)
            model.sim_settings[Parameters.SimulationSetting.AMBIENTTEMPERATURE] = ambient_temperature_data['T'][
                                                                                      0] - 293.15

        # Установка гидравлической корреляции
        if hydr_corr_type.lower() == 'beggsbrill':
            model.sim_settings.global_flow_correlation(
                {Parameters.FlowCorrelation.Multiphase.Vertical.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                 Parameters.FlowCorrelation.Multiphase.Vertical.CORRELATION:
                     constants.MultiphaseFlowCorrelation.BakerJardine.BEGGSBRILLREVISED,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.CORRELATION:
                     constants.MultiphaseFlowCorrelation.BakerJardine.BEGGSBRILLREVISED})
        elif hydr_corr_type == 'Gray':
            model.sim_settings.global_flow_correlation(
                {Parameters.FlowCorrelation.Multiphase.Vertical.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                 Parameters.FlowCorrelation.Multiphase.Vertical.CORRELATION:
                     constants.MultiphaseFlowCorrelation.BakerJardine.GRAY_MODIFIED,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.CORRELATION:
                     constants.MultiphaseFlowCorrelation.BakerJardine.BEGGSBRILLREVISED})
        elif hydr_corr_type == 'gregory':
            model.sim_settings.global_flow_correlation(
                {Parameters.FlowCorrelation.SWAPANGLE: 0,
                 Parameters.FlowCorrelation.Multiphase.Vertical.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.NEOTEC,
                 Parameters.FlowCorrelation.Multiphase.Vertical.CORRELATION:
                     constants.MultiphaseFlowCorrelation.Neotec.GREGORY,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.CORRELATION:
                     constants.MultiphaseFlowCorrelation.BakerJardine.BEGGSBRILLREVISED})
        elif hydr_corr_type.lower() == 'hagedornbrown':
            model.sim_settings.global_flow_correlation(
                {Parameters.FlowCorrelation.SWAPANGLE: 0,
                 Parameters.FlowCorrelation.Multiphase.Vertical.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.TULSA,
                 Parameters.FlowCorrelation.Multiphase.Vertical.CORRELATION:
                     constants.MultiphaseFlowCorrelation.TulsaLegacy.HAGEDORNBROWN_ORIGINAL,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.SOURCE:
                     constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                 Parameters.FlowCorrelation.Multiphase.Horizontal.CORRELATION:
                     constants.MultiphaseFlowCorrelation.BakerJardine.BEGGSBRILLREVISED})
        elif hydr_corr_type.lower() == 'unifiedtuffp':
            model.sim_settings.global_flow_correlation(
                {
                    Parameters.FlowCorrelation.Multiphase.Vertical.SOURCE:
                        constants.MultiphaseFlowCorrelationSource.TUFFPUNIFIED,
                    Parameters.FlowCorrelation.Multiphase.Vertical.CORRELATION:
                        constants.MultiphaseFlowCorrelation.TUFFPUnified.TUFFPV20111_2PHASE,
                    Parameters.FlowCorrelation.Multiphase.Horizontal.SOURCE:
                        constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                    Parameters.FlowCorrelation.Multiphase.Horizontal.CORRELATION:
                        constants.MultiphaseFlowCorrelation.BakerJardine.BEGGSBRILLREVISED})
        elif hydr_corr_type.lower() == 'orkiszewski':
            model.sim_settings.global_flow_correlation(
                {
                    Parameters.FlowCorrelation.Multiphase.Vertical.SOURCE:
                        constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                    Parameters.FlowCorrelation.Multiphase.Vertical.CORRELATION:
                        constants.MultiphaseFlowCorrelation.BakerJardine.DUNSROS,
                    Parameters.FlowCorrelation.Multiphase.Horizontal.SOURCE:
                        constants.MultiphaseFlowCorrelationSource.BAKER_JARDINE,
                    Parameters.FlowCorrelation.Multiphase.Horizontal.CORRELATION:
                        constants.MultiphaseFlowCorrelation.BakerJardine.BEGGSBRILLREVISED})

        return [{Parameters.PTProfileSimulation.INLETPRESSURE: uc.convert_pressure(p, 'pa', 'bar'),
                 Parameters.PTProfileSimulation.LIQUIDFLOWRATE: qliq,
                 Parameters.PTProfileSimulation.FLOWRATETYPE: constants.FlowRateType.LIQUIDFLOWRATE,
                 Parameters.PTProfileSimulation.CALCULATEDVARIABLE: constants.CalculatedVariable.OUTLETPRESSURE},
                heat_balance_pipesim]

    def pipesim_model(self,
                      model,
                      modelcomponents,
                      well_name,
                      gor,
                      wct,
                      dod,
                      gamma_water,
                      gamma_gas,
                      temperature_option,
                      flow_direction=None,
                      h_start=None,
                      **kwargs
                      ):
        # Создадим Black-Oil fluid
        model.add(modelcomponents.BLACKOILFLUID, "BK111",
                  parameters={
                      Parameters.BlackOilFluid.GOR: gor,
                      Parameters.BlackOilFluid.WATERCUT: wct,
                      Parameters.BlackOilFluid.USEDEADOILDENSITY: True,
                      Parameters.BlackOilFluid.DEADOILDENSITY: dod,
                      Parameters.BlackOilFluid.WATERSPECIFICGRAVITY: gamma_water,
                      Parameters.BlackOilFluid.GASSPECIFICGRAVITY: gamma_gas,
                      Parameters.BlackOilFluid.LIVEOILVISCOSITYCORR: "BeggsAndRobinson",
                      Parameters.BlackOilFluid.SinglePointCalibration.SOLUTIONGAS: "Standing",
                      Parameters.BlackOilFluid.SinglePointCalibration.GASCOMPRESSCORRELATION: "Standing",
                      Parameters.BlackOilFluid.SinglePointCalibration.OILFVFCORRELATION: "Standing",
                      Parameters.BlackOilFluid.UNDERSATURATEDOILVISCOSITYCORR: "VasquezAndBeggs"
                  })

        model.add(ModelComponents.SOURCE, "Src-1", \
                  parameters={Parameters.Source.TEMPERATURE: 45,
                              Parameters.Source.PRESSURE: uc.convert_pressure(kwargs['pfl'],
                                                                              'pa',
                                                                              'atm'),
                              Parameters.Source.X_COORD: 98,
                              Parameters.Source.Y_COORD: 144,
                              })

        model.set_value(Source="Src-1", parameter=Parameters.Well.ASSOCIATEDBLACKOILFLUID, value="BK111")

        # create oil sink
        pressure_difference = 0
        if flow_direction == 1:
            pressure_difference = -10
        elif flow_direction == -1:
            pressure_difference = 10
        model.add(ModelComponents.SINK, "Oil-facility", \
                  parameters={Parameters.Sink.PRESSURE: uc.convert_pressure(kwargs['pfl'],
                                                                            'pa',
                                                                            'atm') + pressure_difference,
                              Parameters.Sink.X_COORD: 748,
                              Parameters.Sink.Y_COORD: 312,
                              })

        # create flowline between Src-1 and Oil-facility
        flowline_data = kwargs['flowline_data']
        flowline_inner_diameter = flowline_data['flowline_inner_diameter']
        flowline_length = flowline_data['flowline_length']
        flowline_roughness = flowline_data['flowline_roughness']
        flowline_angle = flowline_data['flowline_angle']
        model.add(ModelComponents.FLOWLINE, "FL-1", \
                  parameters={Parameters.Flowline.DETAILEDMODEL: False,
                              Parameters.Flowline.USEENVIRONMENTALDATA: True,
                              Parameters.Flowline.INNERDIAMETER: flowline_inner_diameter,
                              Parameters.Flowline.LENGTH: flowline_length,
                              Parameters.Flowline.ROUGHNESS: flowline_roughness,
                              Parameters.Flowline.WALLTHICKNESS: 0.5,
                              Parameters.Flowline.UNDULATIONRATE: 0,
                              Parameters.Flowline.HORIZONTALDISTANCE: flowline_length * math.cos(flowline_angle * \
                                                                                                 math.pi / 180),
                              Parameters.Flowline.ELEVATIONDIFFERENCE: flowline_length * math.sin(flowline_angle * \
                                                                                                  math.pi / 180),
                              Parameters.Flowline.AMBIENTTEMPERATURE: kwargs['t1'],
                              Parameters.Flowline.AMBIENTAIRTEMPERATURE: 293.15
                              })

        # Установим temperature option
        # if temperature_option == 'CONST' or temperature_option == 'LINEAR':
        #     model.set_value(Flowline=well_name,
        #                parameter="AmbientTemperature",
        #                value="InputMultipleValues")

        # connect the surface facility into a network
        model.connect({ModelComponents.SOURCE: 'Src-1'}, {ModelComponents.FLOWLINE: 'FL-1'})
        model.connect({ModelComponents.FLOWLINE: 'FL-1'}, {ModelComponents.SINK: 'Oil-facility'})

        # Установим интервал вывода результата
        model.sim_settings[Parameters.SimulationSetting.PIPESEGMENTATIONMAXREPORTINGINTERVAL] = self.L_REPORT

    def save_results(self,
                     uniflocpy_results=None,
                     pipesim_results=None,
                     error_results=None,
                     file_path=None,
                     calc_type: str = 'pipe',
                     equipment_data=None,
                     **kwargs):
        """
        Функция для сохранения результатов в Excel

        Parameters
        ----------
        :param pipesim_results : результаты из Pipesim
        :param file_path : путь к файлу с результатами

        Returns
        -------

        """

        if pipesim_results is None:
            print('Массивы результатов пустые')
            return

        pipesim_results_df = pd.concat(pipesim_results.values(), axis=1)

        # Добавление новых столбцов
        if kwargs['flow_direction'] == 1:
            pipesim_results_df['InclinationAngle'] = kwargs['inclination_angle']
        else:
            pipesim_results_df['InclinationAngle'] = -1. * kwargs['inclination_angle']
        pipesim_results_df['PipeInnerDiameter'] = kwargs['pipe_inner_diameter']
        pipesim_results_df['PipeRelativeRoughness'] = kwargs['pipe_relative_roughness']

        pipesim_output_names = ['InclinationAngle',
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
                                'PressureGradientTotal']
        pipesim_results_df = pipesim_results_df.loc[:, pipesim_output_names]

        with pd.ExcelWriter(file_path) as w:
            pipesim_results_df.to_excel(w)

        pd.DataFrame(pipesim_results_df).to_csv(kwargs['dataset_path'],
                                                mode='a',
                                                header=False,
                                                index=False)

        return pipesim_results_df

    def formate_pipesim_results(self,
                                pipesim_results=None,
                                calc_type: str = 'network',
                                pars=None,
                                equipment_data=None) -> dict:
        """
        Функция для форматирования результатов из Pipesim

        Parameters
        ----------
        :param pars: параметры для вывода из Pipesim
        :param pipesim_results: результаты из Pipesim
        :param calc_type: тип расчета

        Returns
        -------

        """
        if pars is None:
            first_profile_key = list(pipesim_results.profile.keys())[0]
            pipesim_profile = pipesim_results.profile[first_profile_key]
            pars = list(pipesim_profile.keys())
        else:
            pipesim_profile = pipesim_results.profile

        pipesim_results_dict = {}

        if pipesim_results is not None:
            for i in range(len(pars)):
                if pars[i] == ProfileVariables.PRESSURE or pars[i] == ProfileVariables.PRESSURE_GRADIENT_TOTAL or \
                        pars[i] == ProfileVariables.PRESSURE_GRADIENT_ELEVATION or \
                        pars[i] == ProfileVariables.PRESSURE_GRADIENT_FRICTION or \
                        pars[i] == ProfileVariables.PRESSURE_GRADIENT_ACCELERATION:
                    data = uc.convert_pressure(pipesim_profile[pars[i]], 'bar', 'atm')
                elif pars[i] == ProfileVariables.FLOW_PATTERN_GAS_LIQUID:
                    data = pipesim_profile[pars[i]]
                    data = [self.replace_num_flow_pattern(pattr) for pattr in data]
                else:
                    data = pipesim_profile[pars[i]]

                par_df = pd.DataFrame(index=pipesim_profile[ProfileVariables.
                                      MEASURED_DEPTH][:1:-1], data=data[:1:-1],
                                      columns=[pars[i]])

                pipesim_results_dict.update({pars[i]: par_df})
        else:
            print('Результаты Pipesim пустые')

        return pipesim_results_dict

    def main(self,
             fluid_data,
             pipe_data,
             pfl,
             model_path,
             calc_options,
             result_path=None,
             heat_balance=None,
             flow_direction=None,
             h_start=None,
             temperature_option=None,
             ambient_temperature_data: object = None
             ) -> dict:

        # Запуск расчета на Pipesim
        pipesim_results, fluid_data_new, flag = self.calc_model_pipesim(pfl=uc.convert_pressure(pfl, 'atm', \
                                                                                                'pa'),
                                                                        model_path=model_path,
                                                                        heat_balance=heat_balance,
                                                                        fluid_data=fluid_data,
                                                                        pipe_data=pipe_data,
                                                                        temperature_option=temperature_option,
                                                                        ambient_temperature_data=ambient_temperature_data,
                                                                        flow_direction=flow_direction,
                                                                        h_start=h_start
                                                                        )

        pipesim_results_f = self.formate_pipesim_results(pipesim_results=pipesim_results,
                                                         pars=None)

        dataset_path = result_path[:result_path.find('separate_results')] + 'dataset.csv'

        if calc_options['save_results']:
            self.save_results(pipesim_results=pipesim_results_f,
                              file_path=result_path,
                              flow_direction=flow_direction,
                              inclination_angle=pipe_data['flowline_angle'],
                              pipe_inner_diameter=pipe_data['flowline_inner_diameter'],
                              pipe_relative_roughness=pipe_data['flowline_roughness'],
                              dataset_path=dataset_path)

        return {'pipesim_results': pipesim_results_f,
                'density_inversion': flag}
