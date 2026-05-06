import math
import numpy as np
from scipy.integrate import odeint

class UniversalPKSolver:
    def __init__(self, params):
        self.p = params

    def solve(self):
        if self.p.model_type == "1-comp-iv-infusion":
            return self._solve_1comp_iv_infusion()
        elif self.p.model_type == "1-comp-iv-bolus":
            return self._solve_1comp_iv_bolus()
        elif "phenytoin" in self.p.model_type:
            return self._solve_phenytoin() # 페니토인 솔버 호출
        else:
            raise ValueError("Unsupported model type")

    def _solve_phenytoin(self):
        """Phenytoin Michaelis-Menten Kinetics 솔버"""
        # 기본 파라미터 세팅 (Ray Lee 가이드라인 기준)
        wt = self.p.weight
        vd = 0.65 * wt  # Phenytoin 표준 Vd
        v_max = 7.0 * wt # 표준 Vmax (mg/day) -> mg/hr로 환산 필요
        v_max_hr = v_max / 24
        k_m = 4.0 # 표준 Km (mg/L)
        
        # 투여 경로별 S(Salt factor), F(Bioavailability) 설정
        s_factor = 0.92 if "capsule" in self.p.model_type else 1.0
        f_factor = 1.0 # IV 및 경구 완벽 흡수 가정
        
        tau = self.p.tau
        num_doses = self.p.num_doses
        dose = self.p.dose
        tinf = self.p.tinf if self.p.tinf > 0 else 0.5 # Bolus라도 최소 주입시간 설정
        rate = (dose * s_factor * f_factor) / tinf

        # 비선형 미분방정식 정의 (Michaelis-Menten)
        def phenytoin_ode(C, t):
            # 주입 속도 결정
            t_cycle = t % tau
            if t >= tau * (num_doses - 1) + tinf:
                in_rate = 0.0
            elif t_cycle <= tinf:
                in_rate = rate
            else:
                in_rate = 0.0
            
            # dC/dt = (In_rate/Vd) - (Vmax * C / (Vd * (Km + C)))[cite: 6]
            dCdt = (in_rate / vd) - (v_max_hr * C / (vd * (k_m + C)))
            return dCdt

        t_max = tau * num_doses
        t_eval = np.linspace(0, t_max, int(t_max * 10) + 1)
        C_results = odeint(phenytoin_ode, 0.0, t_eval).flatten()

        # 결과 추출
        last_start = int(tau * (num_doses - 1) * 10)
        final_cycle = C_results[last_start:]
        
        return {
            "chart_data": {
                "time": [round(x, 1) for x in t_eval],
                "conc": [round(x, 2) for x in C_results],
                "trough": [round(C_results[i], 2) if i % (tau * 10) == 0 else None for i in range(len(C_results))],
                "ss_trough": None # 비선형은 전통적 SS 수식과 다름
            },
            "metrics": {
                "vd": round(vd, 2),
                "ke": "N/A (Non-linear)",
                "final_peak": round(np.max(final_cycle), 2),
                "final_trough": round(C_results[-1], 2),
                "note": "Michaelis-Menten Kinetics Applied[cite: 6]"
            }
        }

    def _solve_1comp_iv_infusion(self):
        """1-Compartment IV Infusion Model (Vancomycin 표준)"""
        wt, dose, tau, tinf, thalf, num_doses = self.p.weight, self.p.dose, self.p.tau, self.p.tinf, self.p.thalf, self.p.num_doses
        vd = 0.7 * wt[cite: 6]
        ke = math.log(2) / thalf[cite: 6]
        R = dose / tinf[cite: 6]
        
        # 정상상태(SS) 예측치
        cmax_ss = (R / (vd * ke)) * (1 - math.exp(-ke * tinf)) / (1 - math.exp(-ke * tau))
        cmin_ss = cmax_ss * math.exp(-ke * (tau - tinf))
        
        return self._generate_time_series(vd, ke, R, tau, tinf, num_doses, cmax_ss, cmin_ss, "infusion")

    def _solve_1comp_iv_bolus(self):
        """1-Compartment IV Bolus Model (Aminoglycosides 고용량 투여 시)"""
        wt, dose, tau, thalf, num_doses = self.p.weight, self.p.dose, self.p.tau, self.p.thalf, self.p.num_doses
        vd = 0.25 * wt # Gentamicin/Amikacin 표준 Vd 가정
        ke = math.log(2) / thalf
        
        cmax_ss = (dose / vd) / (1 - math.exp(-ke * tau))
        cmin_ss = cmax_ss * math.exp(-ke * tau)
        
        return self._generate_time_series(vd, ke, 0, tau, 0, num_doses, cmax_ss, cmin_ss, "bolus")

    def _generate_time_series(self, vd, ke, R, tau, tinf, num_doses, cmax_ss, cmin_ss, mode):
        """공통 시계열 데이터 생성기"""
        time_data, conc_data, trough_data, ss_trough_data = [], [], [], []
        current_trough = 0
        ss_threshold = cmin_ss * 0.95[cite: 6]
        ss_reached = False

        for n in range(1, num_doses + 1):
            if mode == "infusion":
                peak = (R / (vd * ke)) * (1 - math.exp(-ke * tinf)) + current_trough * math.exp(-ke * tinf)
                next_trough = peak * math.exp(-ke * (tau - tinf))
            else: # Bolus
                peak = (dose / vd) + current_trough
                next_trough = peak * math.exp(-ke * tau)

            if not ss_reached and next_trough >= ss_threshold: ss_reached = True

            for step in range(int(tau * 10) + 1):
                t = step * 0.1
                g_t = (n - 1) * tau + t
                if t > tau + 0.001: continue
                
                if mode == "infusion":
                    c = (R / (vd * ke)) * (1 - math.exp(-ke * t)) + current_trough * math.exp(-ke * t) if t <= tinf else peak * math.exp(-ke * (t - tinf))
                else: # Bolus
                    c = peak * math.exp(-ke * t)
                
                time_data.append(round(g_t, 1))
                conc_data.append(round(c, 2))
                
                if abs(t - tau) < 0.05:
                    trough_data.append(round(next_trough, 2))
                    ss_trough_data.append(round(next_trough, 2) if ss_reached else None)
                elif abs(t) < 0.05 and n == 1:
                    trough_data.append(0.0); ss_trough_data.append(None)
                else:
                    trough_data.append(None); ss_trough_data.append(None)
            
            current_trough = next_trough

        return {
            "chart_data": {"time": time_data, "conc": conc_data, "trough": trough_data, "ss_trough": ss_trough_data},
            "metrics": {"vd": round(vd, 2), "ke": round(ke, 4), "cmax_ss": round(cmax_ss, 2), "cmin_ss": round(cmin_ss, 2), "final_peak": round(peak, 2), "final_trough": round(next_trough, 2)}
        }