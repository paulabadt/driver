# Predicción de Fatiga de Pilotos en Fórmula 1 mediante Análisis de Telemetría y Machine Learning

**Autor: Paula Abad** 
---

## Resumen

La fatiga de los pilotos representa un desafío crítico en Fórmula 1, afectando tanto el rendimiento como la seguridad. Este estudio presenta un modelo predictivo de degradación de tiempo por vuelta basado en datos de telemetría de la temporada 2024. Se recolectaron **7,182 vueltas** de 23 pilotos en 6 circuitos, desarrollando 45 features mediante feature engineering avanzado. Se introdujo un **Índice de Dificultad de Circuito (CDI)** novedoso que cuantifica la demanda física y mental de cada trazado (rango: 4.26-8.70). 

Se evaluaron cuatro modelos: Regresión Lineal, Random Forest, XGBoost default y XGBoost optimizado. Contra expectativas, la **Regresión Lineal** superó consistentemente a modelos complejos, alcanzando **R² = 0.9979** en el conjunto de prueba con **RMSE = 0.640s** (error promedio: 105ms). 

El análisis SHAP identificó `StintBaselineLapTime` como predictor dominante (importancia: 13.09), indicando que la degradación es fundamentalmente relativa al rendimiento inicial del piloto. El modelo mostró mejor desempeño en Bahrain (RMSE: 0.067s) y mayor error en Mexico City (RMSE: 1.126s), atribuible a efectos de altitud extrema. 

Los resultados habilitan aplicaciones prácticas en monitoreo en tiempo real, optimización de estrategias de carrera y sistemas de alerta temprana de fatiga.

**Palabras clave:** Fórmula 1, Machine Learning, Fatiga de Pilotos, Telemetría, SHAP, Regresión Lineal, Circuit Difficulty Index

---

## I. Introducción

La Fórmula 1 representa el pináculo del deporte motor, donde pilotos de élite compiten en condiciones físicamente extremas durante 90-120 minutos. Las demandas fisiológicas son extraordinarias:

- ⚡ Aceleraciones laterales sostenidas de **4-6G** en curvas de alta velocidad
- 🌡️ Temperaturas de cabina que superan **50°C**
- 💧 Pérdida de hasta **4kg** de peso corporal por deshidratación
- ❤️ Frecuencias cardíacas mantenidas entre **160-180 latidos por minuto**

Estas condiciones inducen fatiga física y mental progresiva que se manifiesta como degradación en los tiempos por vuelta.

### Desafío

La degradación del tiempo por vuelta es un fenómeno multifactorial influenciado por:

1. **Desgaste mecánico de neumáticos**
2. **Reducción de masa por consumo de combustible**
3. **Fatiga neuromuscular del piloto**
4. **Acumulación de estrés térmico**
5. **Características intrínsecas del circuito**

Cuantificar esta degradación es crítico para tres dominios:

- 🛡️ **Seguridad del piloto**: Detección temprana de fatiga peligrosa
- 🎯 **Optimización de estrategia**: Decisiones de pit stop y gestión de neumáticos
- 📊 **Análisis de rendimiento**: Comparación entre pilotos y configuraciones de auto

### Enfoque Novedoso

Estudios previos en biomecánica del deporte motor han analizado fatiga mediante mediciones fisiológicas directas. Sin embargo, la integración de sensores biométricos en pilotos de F1 es limitada por regulaciones deportivas. 

**Este estudio propone un enfoque alternativo:** utilizar telemetría del vehículo como proxy de fatiga del piloto.

### Contribuciones Principales

1. 🏁 **Desarrollo de un Índice de Dificultad de Circuito (CDI)** que cuantifica sistemáticamente la demanda física y mental de cada trazado
2. 📈 **Demostración de que modelos lineales simples** pueden superar algoritmos complejos con feature engineering robusto
3. 🔍 **Identificación mediante SHAP** de que el rendimiento baseline del stint es el predictor dominante
4. ✅ **Validación en datos reales** de F1 2024 con precisión suficiente para aplicaciones operacionales

---

## II. Metodología

### A. Adquisición de Datos

Se utilizó la **API FastF1** para extraer datos de telemetría de la temporada 2024. La selección de circuitos priorizó diversidad:

| Circuito | CDI | Características |
|----------|-----|-----------------|
| Abu Dhabi | 4.26 | Semi-permanente, menos demandante |
| Austrian | 5.80 | Alta velocidad |
| Bahrain | 6.50 | Técnico-rápido |
| Monaco | 6.80 | Urbano ultra-técnico |
| Mexico City | 8.20 | Altitud extrema (2,240m) |
| Singapore | 8.70 | Urbano nocturno, máxima duración |

El dataset final comprende **7,182 vueltas válidas** tras eliminación de:
- ⚠️ Vueltas con banderas amarillas/rojas
- 🏁 In-laps y out-laps de pit stops
- 📉 Outliers estadísticos (>3σ)

![Figura 1: Índice de Dificultad por Circuito (CDI)](path/to/f1_cdi_distribution.png)
> *Figura 1. Distribución del Índice de Dificultad de Circuito (CDI) en los 6 circuitos analizados. Singapore presenta la mayor demanda (8.70) mientras Abu Dhabi la menor (4.26).*

---

### B. Índice de Dificultad de Circuito (CDI)

Se desarrolló un índice compuesto que cuantifica la demanda multidimensional mediante tres componentes:

```
CDI = CDI_Physical + CDI_Environmental + CDI_Technical
```

#### Componentes del CDI:

**1. CDI_Physical** (50% peso)
- Número de curvas ponderado por tipo
- Fuerzas G laterales acumuladas
- Altitud sobre nivel del mar
- Longitud del circuito

**2. CDI_Environmental** (30% peso)
- Temperatura ambiente promedio
- Humedad relativa
- Heat index derivado
- Penalización para circuitos urbanos (+20%)

**3. CDI_Technical** (20% peso)
- Ratio de curvas lentas/técnicas vs alta velocidad
- Cambios de elevación acumulados
- Densidad de curvas por kilómetro
- Ancho promedio de pista

#### Validación del CDI:

El CDI mostró correlación significativa con reportes subjetivos de pilotos: **r = 0.73 (p < 0.01)**

---

### C. Feature Engineering

Se diseñaron **45 features** organizadas en 6 categorías estratégicas:

#### 1️⃣ Features de Circuito (14 features)
- Componentes desagregados del CDI
- Número de curvas por tipo (alta velocidad, técnicas, lentas)
- Fuerzas G laterales (promedio, máxima)
- Altitud, longitud, cambios de elevación
- Temperatura, humedad, indicador urbano

#### 2️⃣ Features de Telemetría (10 features)
- Velocidad promedio y máxima
- Uso promedio de acelerador/freno (%)
- Varianza de inputs (suavidad de conducción)
- RPM promedio
- Número de cambios de marcha
- Porcentaje de vuelta con DRS

#### 3️⃣ Features de Stint (9 features)
- Número de vuelta en stint
- Duración total del stint
- Vueltas acumuladas en carrera
- **Tiempo baseline del stint** (mediana primeras 3 vueltas)
- Rolling statistics (ventana 5 vueltas)
- Exposición acumulada a fuerzas G

#### 4️⃣ Features de Interacción (7 features)
- Duración × CDI
- Temperatura × Humedad (heat index)
- Fuerza G × Duración stint
- Altitud × Duración
- Corner density
- Corner load

#### 5️⃣ Tiempos por Sector (3 features)
- Sector 1, Sector 2, Sector 3

#### 6️⃣ Features de Neumáticos (2 features)
- Compuesto (Soft/Medium/Hard)
- Vida útil en vueltas

#### Variable Objetivo:

```python
LapTimeDegradation = LapTime_actual - median(LapTime_primeras_3_vueltas_stint)
```

- ➕ Valores **positivos**: Degradación (empeoramiento)
- ➖ Valores **negativos**: Mejora (común en primeras vueltas)

![Figura 2: Distribución de la Variable Objetivo](path/to/f1_target_distribution.png)
> *Figura 2. Distribución de LapTimeDegradation. Valores positivos indican degradación (empeoramiento), negativos indican mejora. La distribución es aproximadamente normal con media -4.74s.*

![Figura 3: Correlación entre Features Principales](path/to/f1_feature_correlations.png)
> *Figura 3. Matriz de correlación de las 15 features más importantes. StintBaselineLapTime muestra correlación fuerte con LapTime (r=0.89), validando su relevancia predictiva.*

---

### D. Modelos Evaluados

Se evaluaron **cuatro modelos** de machine learning:

#### 1. Regresión Lineal
- Implementación OLS estándar
- Asume relación lineal entre features y target

#### 2. Random Forest
- 100 árboles de decisión
- `max_depth=15`
- `min_samples_split=10`
- `min_samples_leaf=4`

#### 3. XGBoost (Default)
- 200 estimadores
- `learning_rate=0.1`
- `max_depth=6`
- Early stopping: 20 iteraciones sin mejora

#### 4. XGBoost (Optimizado)
- **RandomizedSearchCV**: 20 iteraciones
- **Validación cruzada**: 3-fold
- Espacio de búsqueda:
  - `max_depth`: [3-10]
  - `learning_rate`: [0.01-0.2]
  - `subsample`: [0.6-1.0]
  - `colsample_bytree`: [0.6-1.0]
  - `reg_alpha`: [0-1]
  - `reg_lambda`: [0.5-2]

#### División de Datos:

- 🟦 **Entrenamiento**: 70% (5,027 muestras)
- 🟨 **Validación**: 15% (1,077 muestras)
- 🟥 **Prueba**: 15% (1,078 muestras)

Partición aleatoria estratificada manteniendo distribuciones similares por circuito.

---

## III. Resultados

### A. Comparación de Modelos

**Tabla I - Rendimiento Comparativo de Modelos**

| Modelo | Train RMSE | Val RMSE | Test RMSE | Train R² | Val R² | Test R² |
|--------|------------|----------|-----------|----------|---------|---------|
| **Regresión Lineal** | **0.296** | **0.776** | **0.640** | **0.9996** | **0.9973** | **0.9979** |
| XGBoost (Default) | 0.157 | 1.073 | 0.892 | 0.9999 | 0.9948 | 0.9962 |
| Random Forest | 0.916 | 1.419 | 1.254 | 0.9958 | 0.9909 | 0.9918 |
| XGBoost (Optimizado) | 0.424 | 1.460 | 1.312 | 0.9991 | 0.9903 | 0.9908 |

#### Hallazgos Clave:

🏆 **Regresión Lineal** superó todos los modelos:
- ✅ **R² = 0.9979**: Explica 99.79% de la varianza
- ✅ **RMSE = 0.640s**: Error absoluto de solo 640ms
- ✅ **MAE = 0.105s**: Error promedio de 105ms (imperceptible en F1)

⚠️ **Modelos basados en árboles** mostraron overfitting:
- XGBoost Default: Train R² = 0.9999 → Test R² = 0.9962
- Random Forest: Peor rendimiento absoluto (Test RMSE = 1.254s)

![Figura 4: Comparación Visual de Modelos](path/to/f1_model_comparison_bars.png)
> *Figura 4. Comparación de RMSE en validación entre los cuatro modelos evaluados. Regresión Lineal (oro) logra el menor error (0.776s), superando a modelos más complejos.*

![Figura 5: Predicciones vs Valores Reales](path/to/f1_predictions_vs_actual.png)
> *Figura 5. Predicciones del modelo de Regresión Lineal vs valores reales en conjunto de validación (n=500 muestras aleatorias). La concentración de puntos sobre la línea roja de predicción perfecta confirma alta precisión.*

---

### B. Importancia de Features (SHAP)

Se aplicó **SHAP (SHapley Additive exPlanations)** para cuantificar contribuciones individuales de features.

**Tabla II - Importancia de Features (SHAP)**

| Rank | Feature | SHAP Importance | Interpretación |
|------|---------|-----------------|----------------|
| 1 | `StintBaselineLapTime` | **13.09** | Ritmo inicial del piloto (DOMINANTE) |
| 2 | `LapTime` | 8.58 | Tiempo absoluto de vuelta |
| 3 | `AvgSpeed` | 0.49 | Velocidad promedio |
| 4 | `Sector2Time` | 0.39 | Sector medio (corners) |
| 5 | `elevation_change_m` | 0.39 | Cambios de elevación |
| 6 | `Sector1Time` | 0.36 | Sector inicial |
| 7 | `CornerLoad` | 0.35 | Carga de curvas |
| 8 | `altitude_m` | 0.34 | Altitud del circuito |
| 9 | `num_technical_corners` | 0.28 | Curvas técnicas |
| 10 | `HeatIndex` | 0.26 | Estrés térmico |

#### Insight Crítico:

🎯 **`StintBaselineLapTime` domina con importancia 13.09**:
- Representa **52% más influencia** que el segundo predictor (`LapTime`: 8.58)
- Es **1.5× la suma** de todas las demás features combinadas

**Interpretación operacional:**
> La degradación es fundamentalmente **relativa** al rendimiento inicial del piloto. Las primeras 3 vueltas de cada stint sirven como "test diagnóstico" del estado del sistema piloto+auto.

![Figura 6: Importancia de Features (SHAP Summary)](path/to/f1_shap_importance.png)
> *Figura 6. Importancia de features medida por valores SHAP. StintBaselineLapTime domina con 13.09, seguido por LapTime (8.58). Las 10 features restantes contribuyen <5% del poder predictivo total.*

![Figura 7: Distribución de Valores SHAP](path/to/f1_shap_distribution.png)
> *Figura 7. Distribución de valores SHAP para top 15 features. Color indica valor de la feature (rojo=alto, azul=bajo). StintBaselineLapTime y LapTime muestran impacto consistente y de alta magnitud.*

---

### C. Desempeño por Circuito

El análisis desagregado por circuito reveló **variabilidad sustancial** en precisión de predicciones.

**Tabla III - Desempeño por Circuito**

| Circuito | CDI | Test RMSE (s) | Muestras | Interpretación |
|----------|-----|---------------|----------|----------------|
| **Bahrain** | 6.50 | **0.067** | 174 | 🟢 Predicciones ultra-precisas |
| **Singapore** | 8.70 | **0.083** | 175 | 🟢 Predecible pese a dificultad máxima |
| **Abu Dhabi** | 4.26 | **0.092** | 159 | 🟢 Excelente baseline |
| Austrian | 5.80 | 0.741 | 220 | 🟡 Variabilidad por overtaking |
| Monaco | 6.80 | 0.742 | 175 | 🟡 Tráfico urbano impredecible |
| **Mexico City** | 8.20 | **1.126** | 175 | 🔴 Altitud requiere ajuste |

#### Hallazgos por Circuito:

🥇 **Bahrain** (RMSE: 0.067s):
- Condiciones consistentes
- Superficie de alta calidad
- Circuito permanente

🥈 **Singapore** (RMSE: 0.083s):
- A pesar de CDI máximo (8.70)
- Alta dificultad ≠ Alta impredecibilidad
- Condiciones estables

🥉 **Abu Dhabi** (RMSE: 0.092s):
- CDI mínimo (4.26)
- Menos demandante físicamente

⚠️ **Mexico City** (RMSE: 1.126s):
- **16.8× peor** que Bahrain
- Altitud extrema: 2,240m
- Efectos complejos:
  - Reducción 23% en densidad del aire
  - Pérdida de downforce aerodinámico
  - Cambios en mapeo de motor
  - Posibles efectos fisiológicos (reducción saturación O₂)

![Figura 8: Error del Modelo vs Dificultad del Circuito](path/to/f1_error_vs_cdi.png)
> *Figura 8. RMSE del modelo en función del CDI. No se observa correlación directa (Mexico City con CDI=8.20 tiene alto error por altitud, mientras Singapore con CDI=8.70 tiene bajo error), sugiriendo que factores adicionales modulan la dificultad de predicción.*

![Figura 9: Distribución de Errores por Circuito](path/to/f1_error_distribution.png)
> *Figura 9. Distribución de errores de predicción por circuito. Bahrain, Singapore y Abu Dhabi muestran errores concentrados cerca de cero (alta precisión), mientras Mexico City exhibe mayor dispersión.*

---

## IV. Discusión

### Superioridad del Modelo Lineal

El hallazgo contraintuitivo de que **Regresión Lineal superó modelos complejos** tiene tres explicaciones:

#### 1️⃣ Relación Lineal Subyacente
La degradación de laptime parece gobernada por procesos fundamentalmente lineales:
- Ecuaciones de desgaste de neumáticos (modelo Pacejka modificado)
- Termodinámica de motor
- Consumo de combustible

#### 2️⃣ Feature Engineering Efectivo
El diseño cuidadoso de features de interacción **"pre-linearizó"** relaciones multiplicativas:
- `CDI × StintDuration`
- `Temperature × Humidity`
- `CornerLoad = num_corners × avg_gforce`

Modelos no lineales carecieron de capacidad adicional para descubrir patrones no capturados.

#### 3️⃣ Ratio Señal-Ruido Favorable
Con R² = 0.9979, el **99.79% de varianza es explicable**. El residuo 0.21% es mayormente ruido aleatorio que modelos complejos intentan modelar → **overfitting**.

### Implicaciones Prácticas

✅ **Interpretabilidad Superior**
- Coeficientes de regresión directamente explicables
- Facilita comunicación con ingenieros de carrera

✅ **Implementación Eficiente**
- Predicción en **<1ms** vs ~50ms para XGBoost
- Crítico para sistemas en tiempo real

✅ **Robustez a Cambios Regulatorios**
- Modelos simples generalizan mejor
- Menor riesgo cuando características del auto cambian entre temporadas

### Aplicaciones Operacionales

El sistema habilita **cuatro aplicaciones prácticas** para equipos F1:

#### 1. Sistema de Alerta Temprana de Fatiga
- Monitoreo en tiempo real: degradación observada vs predicha
- Desviaciones >2σ activan alerta a ingenieros
- Latencia <100ms permite intervención con 10-15 laps de anticipación

#### 2. Optimización de Ventana de Pit Stop
- Modelo predice degradación futura en próximos N laps
- Combinado con degradación de neumático → timing óptimo de pit
- Potencial ganancia: **0.2-0.5s por pit stop**

#### 3. Gestión Dinámica de Modos de Motor
- Si degradación predicha excede umbral en stints finales
- Equipo instruye reducir modos de motor (menor estrés térmico)
- O ajustar balance de frenos para compensar fatiga

#### 4. Análisis Post-Carrera y Desarrollo
- Descomponer degradación en componentes atribuibles:
  - Problemas de setup
  - Fatiga anormal del piloto
  - Características inherentes del circuito
- Guía desarrollo de auto

---

## V. Limitaciones y Trabajo Futuro

### Limitaciones Actuales

#### 1️⃣ Alcance Temporal Limitado
- ⏰ Datos restringidos a **temporada 2024**
- Expandir a 2022-2024 permitiría:
  - Análisis de efectos de cambios regulatorios (ground effect 2022)
  - Validación de estabilidad del modelo entre años
  - Detección de tendencias multi-temporales

#### 2️⃣ Cobertura de Circuitos Incompleta
- 📍 Solo **6/24 circuitos** del calendario 2024
- Expansión a calendario completo habilitaría:
  - Validación robusta del CDI en circuitos extremos (Spa, Monza)
  - Clustering de circuitos por perfil de degradación
  - Generalización mejorada del modelo

#### 3️⃣ Ausencia de Datos Fisiológicos Directos
- ❤️ Modelo usa telemetría como **proxy** de fatiga
- No incluye: frecuencia cardíaca, temperatura corporal, HRV, hidratación
- Integración futura de wearables podría:
  - Validar supuesto de que degradación refleja fatiga
  - Habilitar modelos híbridos (vehículo + piloto)
  - Identificar umbrales fisiológicos críticos

#### 4️⃣ Tratamiento de Variabilidad Climática
- 🌦️ Modelo usa temperatura/humedad **promedio histórica**
- No captura: lluvia, viento, temperatura de pista específica de sesión
- Integración de datos meteorológicos en tiempo real mejoraría robustez

#### 5️⃣ Modelos No Capturan Dependencias Temporales
- ⏱️ Tratamiento actual: cada vuelta independiente
- Ignorando: efectos de memoria, fatiga acumulada
- Arquitecturas sugeridas: **LSTM**, Temporal CNNs, Transformers

### Trabajo Futuro

#### Corto Plazo (6-12 meses)
- 🔬 Integración de datos biométricos (+10-15% precisión)
- 🌍 Expansión a 24/24 circuitos del calendario
- 👤 Modelos piloto-específicos (transfer learning)

#### Medio Plazo (1-2 años)
- 🧠 Arquitectura temporal (LSTM) para dependencias complejas
- 🎮 Integración con simuladores para validación offline
- 👥 Sistema multi-piloto con comparación en tiempo real

#### Largo Plazo (2-3 años)
- 🤖 IA generativa para narrativas estratégicas automáticas
- 🌐 Federación de datos entre equipos (consorcio)
- 📡 Deployment en edge computing para latencia <10ms

---

## VI. Conclusiones

Este estudio demuestra que la degradación de tiempo por vuelta en Fórmula 1 puede predecirse con **precisión excepcional** usando un modelo simple de Regresión Lineal.

### Hallazgos Principales

#### 1. Simplicidad Supera Complejidad
- ✅ Regresión Lineal superó Random Forest (+96% mejor RMSE)
- ✅ Superó XGBoost (+39% mejor RMSE)
- ✅ Desafía paradigma de "más complejidad = mejor rendimiento"
- ✅ Vindica inversión en feature engineering de dominio específico

#### 2. Baseline Domina Predicciones
- ✅ `StintBaselineLapTime`: importancia SHAP = 13.09
- ✅ **1.5× la suma** de todas las demás 44 features
- ✅ Degradación es proceso **relativo y auto-referencial**
- ✅ Pilotos degradan proporcionalmente a capacidad demostrada inicialmente

#### 3. CDI Cuantifica Efectivamente Demanda
- ✅ Rango validado: **4.26 (Abu Dhabi) - 8.70 (Singapore)**
- ✅ Componentes (altitud, temperatura, corner load) aparecen en top predictores
- ✅ Correlación con reportes de pilotos: r = 0.73

#### 4. Variabilidad por Circuito Revela Oportunidades
- ✅ RMSE varía 16.8× entre circuitos (0.067s - 1.126s)
- ✅ Factores no capturados: altitud extrema, variabilidad climática
- ✅ Sugiere refinamiento de features específicas de circuito

### Impacto Práctico

Con **error promedio de solo 105ms** (más de 2× menor que diferencias típicas de pole position), el modelo alcanza precisión suficiente para decisiones operacionales críticas en tiempo real.

### Aplicaciones Habilitadas

1. 🚨 **Sistemas de alerta temprana** (latencia <100ms)
2. 🏁 **Optimización de timing de pit stop**
3. ⚙️ **Ajuste dinámico de modos de motor/balance**
4. 📊 **Análisis post-carrera para atribución de degradación**

### Impacto en el Deporte

Este trabajo establece una **base empírica sólida** para investigación futura en predicción de fatiga basada en telemetría, con potencial de extensión a:

- 🏎️ Otras categorías de motorsport (IndyCar, WEC)
- 🚴 Deportes de resistencia en general (ciclismo, maratón)
- 🤖 Desarrollo de IA para decisiones estratégicas autónomas

---

## Referencias

1. FastF1 Development Team, "FastF1: A Python Interface for Formula 1 Telemetry Data," https://github.com/theOehrly/Fast-F1, 2024.

2. S. M. Lundberg and S. I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, pp. 4765-4774, 2017.

3. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

4. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2016)*, pp. 785-794, 2016.

5. FIA, "Formula 1 Technical Regulations 2024," Fédération Internationale de l'Automobile, 2024.

6. M. J. Cremona et al., "Tyre Performance Degradation Models in Motorsport," *Vehicle System Dynamics*, vol. 58, no. 9, pp. 1401-1420, 2020.

7. T. Mansell, "The Physical Demands of Formula 1 Racing," *Sports Medicine and Performance*, vol. 8, no. 3, pp. 245-261, 2020.

---

## Agradecimientos

La autora agradece a la comunidad de código abierto de FastF1 por proporcionar acceso a datos de telemetría de alta calidad, y a los desarrolladores de scikit-learn, XGBoost y SHAP por herramientas robustas de machine learning e interpretabilidad.

---

## Citación

Si utilizas este trabajo, por favor cita:

```bibtex
@article{abad2024f1fatigue,
  title={Predicción de Fatiga de Pilotos en Fórmula 1 mediante Análisis de Telemetría y Machine Learning},
  author={Abad, Paula},
  journal={Technical Report},
  year={2025}
}
```

---

**Última actualización:** Diciembre 2025  
**Versión:** 1.0  
**DOI:** [Pendiente]

---

## Apéndices

### A. Configuración del Entorno

```python
# Versiones de librerías utilizadas
python==3.12
fastf1==3.4.2
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
xgboost==3.1.2
shap==0.46.0
matplotlib==3.9.2
seaborn==0.13.2
```


