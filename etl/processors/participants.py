import pandas as pd
import numpy as np
from etl.processors.base import BaseDataProcessor

class ParticipantProcessor(BaseDataProcessor):
    """
    Обработка ВСЕХ людей: в машинах (vehicles) и снаружи (participants).
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.mode = self.config.get('violations_mode', 'pre_event')

    def _classify_violations(self, violations: list) -> dict:
        """
        Классификация нарушений (Водители + Пешеходы).
        """
        flags = {
            'viol_priority': 0,   # Приоритет (Водители)
            'viol_pedestrian': 0, # Вина пешехода (Переход, Игра, Выход из-за ТС)
            'viol_speed': 0,      # Скорость (Водители)
            'viol_drunk': 0,      # Пьяный (Водитель или Пешеход)
            'viol_runaway': 0,    # Побег (Утечка)
            'viol_safety': 0,     # Ремни, Шлемы, Световозвращатели
            'viol_rights': 0,     # Нет прав
            'viol_admin': 0       # Бюрократия
        }
        
        if not violations: return flags

        for v in violations:
            s = self.normalize_text(str(v))
            
            # 1. Пьянство (Водитель или Пешеход)
            if any(x in s for x in ['опьянен', 'отказ', 'наркотич', 'алкогол']):
                flags['viol_drunk'] = 1
            
            # 2. Вина пешехода (Специфика participants)
            if any(x in s for x in [
                'переход', 'пешеход', 'проезж', 'неожиданный', 'игра_на', 
                'скейтборд', 'роликов', 'ходьба_вдоль'
            ]):
                flags['viol_pedestrian'] = 1
                
            # 3. Безопасность (Ремни + Световозвращатели)
            if any(x in s for x in ['ремень', 'шлем', 'детск', 'кресл', 'неисправн', 'световозвращ']):
                flags['viol_safety'] = 1
            
            # 4. Водительские нарушения
            if any(x in s for x in ['очередност', 'встречн', 'светофор', 'регулировщик', 'перестроени']):
                flags['viol_priority'] = 1
            if any(x in s for x in ['скорост', 'дистанц', 'интервал']):
                flags['viol_speed'] = 1
            if 'оставление' in s:
                flags['viol_runaway'] = 1
            if any(x in s for x in ['не_имеющ', 'лишен', 'категори']):
                if 'документ' not in s: 
                    flags['viol_rights'] = 1
            if any(x in s for x in ['осаго', 'документ']):
                flags['viol_admin'] = 1
                    
        return flags

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Выбор колонок нарушений
        target_viols = []
        if self.mode == 'pre_event':
            target_viols = ['viol_priority', 'viol_pedestrian', 'viol_speed', 'viol_drunk', 'viol_safety', 'viol_rights']
        elif self.mode == 'post_event':
            target_viols = ['viol_priority', 'viol_pedestrian', 'viol_speed', 'viol_drunk', 'viol_safety', 'viol_rights', 'viol_runaway', 'viol_admin']
        
        with self.activity(f"Обработка участников (Режим нарушений: {self.mode})"):
            
            res = {
                'ppl_count': [], 
                'ppl_pass_count': [], 
                'ppl_ped_count': [], 
                'ppl_solo_count': [],
                'ppl_kids_count': [],
                
                # Типы пешеходов (Новое!)
                'ppl_ped_worker_count': [], # Рабочие на дороге
                'ppl_ped_other_count': [],
                
                'ppl_drv_male': [], 'ppl_drv_female': [],
                'ppl_drv_exp_min': [], 'ppl_drv_exp_mean': [], 'ppl_drv_novice': []
            }
            for v in target_viols: res[v] = []

            # Итераторы (безопасное получение)
            peds_series = df['participants'].fillna("").apply(lambda x: x if isinstance(x, list) else []) if 'participants' in df.columns else [[]]*len(df)
            vehicles_series = df['vehicles'].fillna("").apply(lambda x: x if isinstance(x, list) else [])

            for v_list, p_list in zip(vehicles_series, peds_series):
                
                # Локальные счетчики
                cnt_total = 0
                cnt_pass = 0
                cnt_ped = 0
                cnt_ped_worker = 0
                cnt_ped_other = 0
                
                cnt_solo = 0
                cnt_kids = 0
                
                cnt_male = 0
                cnt_female = 0
                cnt_novice = 0
                exps = []
                
                row_viols = {k: 0 for k in target_viols}
                
                # --- 1. ЛЮДИ В МАШИНАХ ---
                for car in v_list:
                    parts = car.get('participants', [])
                    if not isinstance(parts, list): parts = []
                    
                    p_count = len(parts)
                    cnt_total += p_count
                    if p_count == 1: cnt_solo += 1
                    
                    for p in parts:
                        role = self.normalize_text(p.get('role', ''))
                        gender = self.normalize_text(p.get('gender', ''))
                        cat = self.normalize_text(p.get('category', ''))
                        
                        if 'пассажир' in role: cnt_pass += 1
                        if 'ребенок' in role or 'дети' in cat: cnt_kids += 1
                        
                        if 'водитель' in role:
                            if 'муж' in gender: cnt_male += 1
                            elif 'жен' in gender: cnt_female += 1
                            try:
                                e = float(p.get('years_of_driving_experience'))
                                if 0 <= e <= 70:
                                    exps.append(e)
                                    if e < 2: cnt_novice += 1
                            except: pass
                        
                        if self.mode != 'skip':
                            v_flags = self._classify_violations(p.get('violations', []))
                            for k in target_viols:
                                if v_flags.get(k) == 1: row_viols[k] = 1

                # --- 2. ПЕШЕХОДЫ (Вне машин) ---
                cnt_ped = len(p_list)
                cnt_total += cnt_ped
                
                for p in p_list:
                    role = self.normalize_text(p.get('role', ''))
                    cat = self.normalize_text(p.get('category', ''))
                    
                    if 'ребенок' in role or 'дети' in cat: cnt_kids += 1
                    if 'до_7_лет' in str(p.get('violations', [])): cnt_kids += 1
                    
                    if any(x in role for x in ['работник', 'сотрудник', 'дпс', 'полици', 'дорожн']):
                        cnt_ped_worker += 1
                    elif any(x in role for x in ['торгов', 'перекрыти', 'иной']):
                        cnt_ped_other += 1
                    
                    if self.mode != 'skip':
                        v_flags = self._classify_violations(p.get('violations', []))
                        for k in target_viols:
                            if v_flags.get(k) == 1: row_viols[k] = 1

                res['ppl_count'].append(cnt_total)
                res['ppl_pass_count'].append(cnt_pass)
                res['ppl_ped_count'].append(cnt_ped)
                res['ppl_ped_worker_count'].append(cnt_ped_worker) # <---
                res['ppl_ped_other_count'].append(cnt_ped_other)   # <---
                res['ppl_solo_count'].append(cnt_solo)
                res['ppl_kids_count'].append(cnt_kids)
                
                res['ppl_drv_male'].append(cnt_male)
                res['ppl_drv_female'].append(cnt_female)
                res['ppl_drv_novice'].append(cnt_novice)
                
                if exps:
                    res['ppl_drv_exp_min'].append(min(exps))
                    res['ppl_drv_exp_mean'].append(round(sum(exps)/len(exps), 1))
                else:
                    res['ppl_drv_exp_min'].append(-1)
                    res['ppl_drv_exp_mean'].append(-1)
                
                for k in target_viols:
                    res[k].append(row_viols[k])

            for k, v in res.items():
                if not v and 'viol_' in k: continue
                df[k] = v
                
            cols_to_drop = ['vehicles', 'participants', 'participant_categories', 'participants_count']
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
                
        return df