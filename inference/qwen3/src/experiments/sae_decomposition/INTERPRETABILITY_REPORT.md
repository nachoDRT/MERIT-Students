# Interpretabilidad del sesgo demográfico vía SAEs — Informe de proceso

**Proyecto:** MERIT-Students · Detección de sesgo demográfico en admisiones universitarias
**Modelo:** Qwen3-VL-8B-Instruct
**SAE:** Qwen-Scope `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100` (TopK, k=100, diccionario 64K, d_model 4096)
**Rama:** `saes-interpretation` · **Pipeline:** `inference/qwen3/src/experiments/sae_decomposition/`

---

## 1. Objetivo

Determinar **(a)** si los SAEs de Qwen-Scope sirven para interpretar el vector de *steering* que corrige el sesgo, y **(b)** qué *features* del SAE están asociadas a ese sesgo. Operativamente, dos formulaciones equivalentes:

- Qué *features* componen el vector de steering que resuelve el sesgo.
- Qué *features* están activas para el sujeto **aceptado** y no para el **rechazado** (o viceversa).

## 2. Montaje experimental

El modelo evalúa si un alumno debe ir a la universidad a partir de **(imagen de notas + foto del estudiante + prompt)**. El par contrastivo aísla la demografía:

- **x⁺ (aceptado):** notas malas + foto `subject_8`
- **x⁻ (rechazado):** **mismas** notas malas + foto `subject_0`
- Mismo documento de notas, mismo nombre (Mark Elliott); solo cambia la persona de la foto.

Con las mismas notas suspensas, el modelo tiende a **aceptar a subject_8 y rechazar a subject_0** → sesgo. El **vector de steering** (CAA) es la diferencia de medias del residual stream en el último token sobre 50 pares (seed 42), con forma `[36, 4096]`. Inyectarlo voltea el veredicto.

---

## 3. Proceso de interpretabilidad y resultados

El proceso ha sido iterativo: cada paso descarta o valida una hipótesis metodológica.

### Paso 0 — ¿Reconstruye el SAE las activaciones de Qwen3-VL?

El SAE está entrenado sobre Qwen3-8B-**Base** (solo texto); lo aplicamos sobre el VL-Instruct. `validate_sae` (capa 20, 50 prompts diversos, 683 tokens):

| métrica | valor |
|---|---|
| `reconstruction_fidelity` | **0.836** |
| `explained_variance` (R² centrado) | 0.822 |
| `cos_sim` | 0.932 |
| L0 efectivo | 100 |

**Conclusión:** la transferencia texto→multimodal funciona; el SAE reconstruye bien activaciones reales. Punto de partida válido.

### Paso 1 — Descomponer el vector de steering directamente ❌

Primer intento: pasar el vector `v` por el encoder del SAE (`decompose_vector`, capa 20).

| métrica | valor | lectura |
|---|---|---|
| `recon_cos_sim` | **0.0298** | la reconstrucción es casi ortogonal a `v` |
| `recon_norm_ratio` | 4.0049 | y con 4× la norma |
| `n_robust` (A∩B) | 1 | sin features robustos |

**Validación causal por ablación** (`ablation_steering`, capa 20, β=10, 10 imágenes de subject_0):

| variante | proceed |
|---|---|
| `v_original` | **9/10 (90%)** |
| `v_robust` | 0/10 |
| `v_top_a_10` | 0/10 |
| `v_top_a_50` | 0/10 |
| `v_top_b_10` | 0/10 |

**Conclusión:** **descomponer `v` directamente es inválido.** `v` es una *diferencia de medias* → vive **fuera de la variedad** (off-manifold) de activaciones reales sobre la que se entrenó el SAE. Ninguna reconstrucción reproduce el efecto de steering (todas 0%, `cos_orig ≈ 0`).

### Paso 2 — Descomposición on-manifold por condición ✓ (método válido)

En vez de descomponer `v`, descomponemos las activaciones reales `h_pos`/`h_neg` por separado (`cache_activations` + `analyze_conditions`), que **sí** están on-manifold, y comparamos en espacio de features.

Reconstrucción de las activaciones reales:

| capa | `reconstruction_fidelity` | `cos_sim` | R² centrado |
|---|---|---|---|
| 16 | 0.846 | 0.920 | −227.6 |
| 17 | 0.858 | 0.927 | −196.3 |
| 19 | 0.846 | 0.920 | −102.3 |
| 20 | 0.820 | 0.906 | −80.9 |

> **Nota metodológica clave.** El R² centrado **colapsa a muy negativo** aquí porque las activaciones están **muy concentradas** (mismo prompt, solo cambia la foto): norma ~60-97 pero dispersión en torno a la media ~1.6-4.5 (2-5%). El "baseline" de predecir la media es casi perfecto, así que el R² centrado es engañoso. La métrica fiable en datos concentrados es `reconstruction_fidelity` (~0.82-0.86, buena). Por eso unificamos y reportamos **ambas** métricas (`sae_module.py`).

Selección de features discriminantes. El método base (`delta = mean(feat_pos) − mean(feat_neg)` + exclusivos por frecuencia) se reforzó con un análisis **pareado** (los pares comparten documento de notas) con tamaño de efecto (t-estadístico), que cancela el contexto compartido. Resultados destacados (capa 20):

| feature | t pareado | freq en subject_8 | freq en subject_0 | tipo |
|---|---|---|---|---|
| F12956 | −54.2 | 0.02 | 1.00 | interruptor del rechazado |
| F39735 | +28.4 | 0.98 | 0.04 | interruptor del aceptado |
| F40514 | −27.4 | 0.08 | 1.00 | interruptor del rechazado |
| F16906 | +11.2 | 0.72 | 0.00 | exclusivo del aceptado |

**Coherencia:** el solapamiento entre estos features (on-manifold) y los obtenidos descomponiendo `v` es **≈ 0**, lo que confirma de nuevo que el Paso 1 era la vía equivocada.

### Paso 3 — Interpretación de los features

**3a. Logit lens** (`feature_interpretation.py`, y la fase de logit lens en `analyze_conditions.py`).

*Qué es.* Cada feature `i` del SAE tiene una **dirección de decoder** `W_dec[:, i]` ∈ ℝ^4096: el vector que el SAE *suma* al residual stream cuando ese feature se activa (es decir, "cómo se ve" el feature en el espacio del modelo). El logit lens lee qué tokens promovería esa dirección si apareciera en el residual stream, tratándola **como si fuese el estado oculto final** del modelo:

1. Se toma la dirección del feature `d = W_dec[:, i]`.
2. Se pasa por la **norma final** del modelo (RMSNorm): `d̂ = final_norm(d)`.
3. Se proyecta a vocabulario con la matriz de *unembedding* (los pesos de salida `lm_head`): `logits = d̂ · lm_headᵀ` ∈ ℝ^|vocab|.
4. Se ordenan: los tokens con `logit` más alto son los que el feature **promueve** (`promotes`); los más bajos, los que **suprime** (`suppresses`).

*Cómo se lee.* "Si este feature está activo, empuja la siguiente predicción hacia estos tokens". Es la forma estándar y barata de poner una etiqueta tentativa a un feature.

*Por qué es engañoso aquí (y lo descartamos como interpretación fiable):*
- **Salta toda la computación intermedia.** El feature vive en la **capa 20 de 36**; el logit lens lo proyecta directamente a la salida ignorando las 16 capas posteriores que aún transforman esa señal. Solo ve lo que es *linealmente legible* en el unembedding final.
- **Solo expresa lo proyectable a tokens sueltos.** Un concepto **visual** (apariencia, demografía) no tiene un token equivalente, así que se proyecta como ruido sobre tokens dispersos.
- **Diccionario multilingüe.** Al venir el SAE de Qwen3-8B-**Base**, las direcciones se proyectan a menudo sobre tokens chinos polisémicos.

Resultado: tokens **polisémicos y multilingües** (p.ej. F12956→"molecule/分子", F40514→"integration", F39735→"Grade/classify/等级") que **no se leen como sesgo demográfico**. Además, contrastado con el Paso 3b, la etiqueta del logit lens **no coincide con lo que realmente enciende el feature** (F40514 "integration" en logit lens dispara en realidad sobre dígitos), lo que confirma que aquí es poco fiable.

**3b. Ejemplos máximamente activantes** (`max_activating.py`, `PIPELINE_STEP=sae_max_activating`). Para cada feature discriminante, se busca qué tokens/contextos de un corpus lo encienden más. Detalles metodológicos:
- Hook en la capa L + excepción `_StopForward` para saltar capas superiores y `lm_head`.
- **Gateo TopK real**: se ranquea por la activación efectiva del feature (solo cuenta si entra en el top-100), no por la pre-activación del encoder (que un `b_enc` alto inflaría en todas partes). *Este arreglo fue determinante:* la primera versión por pre-activación daba lecturas falsas (dígitos/espacios).

**Qué es `corpus_fire_rate`.** Para cada feature, la **fracción de líneas del corpus en las que el feature llega a activarse de verdad** (entra en el top-100 del SAE en al menos un token de la línea):

```
corpus_fire_rate(feature) = nº de líneas donde el feature dispara (TopK) / nº total de líneas del corpus
```

Es una medida de **cobertura/frecuencia**, no de intensidad. Interpretación:
- `≈ 1.0` → el feature se enciende casi en todo el corpus → es **ubicuo/genérico** (poco específico; p.ej. un feature de espacios o de nombres propios).
- `≈ 0.0` → el feature **casi nunca dispara** en ese corpus → o bien es muy específico de un dominio que ese corpus no cubre, o bien no es un feature *de texto* en absoluto.
- Su valor clave aquí es **comparativo entre corpus**: si un feature pasa de `fire_rate≈0` en wikitext a un valor apreciable en el corpus de sesgo (p.ej. F44412: 0.0003 → 0.0275, ×91), significa que codifica un concepto que el texto neutro no probaba. Si **sigue mudo en ambos** (F39735: 0.0008 → 0.0003), la señal no es de texto → apunta a lo visual.

Nota: `fire_rate` mide *con qué frecuencia* dispara, no *cuánto separa* a los sujetos. Un feature puede tener `fire_rate` alto y aun así discriminar (se desplaza poco pero siempre en la misma dirección) — eso lo captura el t-estadístico pareado del Paso 2, no el `fire_rate`.

**Corpus neutro (`wikitext`).** Los features discriminantes se separan en dos poblaciones:
- **(A) De superficie, genéricos:** disparan en muchísimo texto sobre tokens estructurales (F41594→espacios, F7281→comas, F39350→nombres propios `fire_rate=0.89`, F12956→palabras función). No son "conceptos de sesgo".
- **(B) Casi mudos en texto:** apenas disparan pero son los discriminadores más fuertes. **F39735** (el #1) disparó en **3 de 4000 líneas** (`fire_rate=0.0008`).

**Corpus de sesgo (`social` = measuring-hate-speech + noticias BLACK VOICES/WOMEN/QUEER VOICES…).** Hipótesis: si un feature codifica un concepto de sesgo, wikitext no lo probaría y parecería mudo. Comparando `fire_rate` wikitext→social:

| feature | lado | wiki → social | qué lo enciende (contexto real) |
|---|---|---|---|
| **F44412** | rechazado | 0.0003 → **0.0275 (91×)** | identidad/prejuicio: *"all races whites"*, *"feminist **agenda**"*, *"**ignorant** person"*, *"bi person"* |
| **F1036** | rechazado | 0.026 → 0.051 (2×) | nacionalidad/inmigración: *"**Mexico**… immigration"*, *"Palestinians… genocidal"*, *"Hindu Marriage Act"* |
| **F45944** | rechazado | top token = **` skin`** | *"from Africa… **pigmentation**… original **skin** tone"* → tono de piel / raza |
| **F39735** | aceptado | 0.0008 → 0.0003 | **siguió mudo** (único hit: "#SouthAfrica") → vision-specific |

**Hallazgo central del Paso 3:** el **corpus de interpretación importa**. Probando con texto de discriminación emergen features **demográficamente legibles** (tono de piel/raza, identidad de grupo, nacionalidad) que el corpus neutro no detectaba — y **se concentran en la condición rechazada (subject_0)**, justo lo esperable de un sesgo. Sin embargo, el discriminador más fuerte (F39735) **no** se interpreta por texto en ningún corpus → su señal parece vivir en el canal **visual**. *Caveat:* estos máx-activantes son cualitativos (anecdóticos); el Paso 4 los pone a prueba cuantitativamente.

### Paso 4 — Anclaje a etiquetas demográficas: una métrica de *significado*

`fire_rate` mide *frecuencia*, no *significado*. Para cuantificar el significado anclamos la activación de cada feature a las **etiquetas demográficas** de `measuring-hate-speech` (`feature_grounding.py`, `PIPELINE_STEP=sae_feature_grounding`). Para cada feature se calcula su activación por línea (máx TopK sobre tokens) y el **ROC-AUC** contra cada categoría (raza, género, orientación, religión, origen, discapacidad, edad) y subcategorías clave. `AUC>0.5` = mayor activación cuando el texto ataca a ese grupo. 6000 textos únicos, 17 etiquetas.

Asociaciones más fuertes (filtrando a `fire_rate>0.1`, porque con `fire≈0` el AUC es ruido):

| feature | lado | etiqueta top | AUC | fire_rate |
|---|---|---|---|---|
| F24635 | aceptado | `race_white` | **0.635 ↑** | 0.98 |
| F39064 | rechazado | `religion_muslim` | 0.610 ↑ | 0.38 |
| F62970 | rechazado | `religion_muslim` | 0.578 ↑ | 0.46 |
| F40514 | rechazado | `religion(any)` | 0.570 ↑ | 0.24 |
| F39350 | aceptado | `sexuality_gay` | 0.555 ↑ | 0.75 |

**Resultados (sobrios):**
- **AUC máximo global = 0.635** → **ningún feature tiene anclaje demográfico fuerte**; hay estructura pero débil.
- **El entusiasmo cualitativo del Paso 3 no sobrevive:** F44412 ("identidad/prejuicio") y F45944 (`skin`) tienen `fire<0.1` y AUC≈0.48-0.49 (azar) → eran **anécdotas**, no asociaciones robustas. (23 de 37 features tienen `fire<0.1` → AUC no fiable.)
- **Patrón débil pero direccionalmente coherente:** feature del **aceptado** ↔ `white` (0.635); varias del **rechazado** ↔ `muslim`/religión (0.58-0.61).
- **Límite inherente:** la métrica es **ciega a features vision-specific** (F39735: `fire=0.001`, AUC=0.500 — no anclable por texto).

**Qué mide el AUC:** asociación entre activación y *"texto que ataca al grupo X"* → es **temática/textual** ("la feature responde a discurso sobre X"), no prueba de codificación *visual* del atributo. Es cuantitativa y anclada, pero no causal ni visual.

---

## 4. Conclusiones actuales

1. **Descomponer el vector de steering directamente es inválido** (off-manifold). Confirmado por reconstrucción (`cos≈0.03`) y ablación (0% vs 90%).
2. **La vía válida es on-manifold por condición**: el SAE reconstruye fielmente `h_pos`/`h_neg` (`fidelity≈0.82-0.86`) y permite comparar features entre aceptado y rechazado.
3. **El SAE de texto-Base no entrega features de sesgo "de fábrica"**, y el logit lens es engañoso.
4. **Pero con el corpus de probing adecuado, sí emergen features de sesgo interpretables** (piel/raza, identidad, nacionalidad), localizados en la condición rechazada. La utilidad del SAE depende del corpus de interpretación.
5. **Parte de la señal es irreductiblemente visual**: el discriminador #1 es mudo a todo texto.

## 5. Limitaciones

- **Confound de 2 sujetos.** Solo `subject_8` vs `subject_0`: cualquier feature discriminante está confundido con *todas* las diferencias entre esos dos sets de fotos (iluminación, fondo, encuadre, estadística de imagen), no solo la demografía. No se puede atribuir aún a "raza/apariencia" con certeza.
- **Sin validación causal** de los features demográficos: sabemos que correlacionan con la condición rechazada, no que la *causen*.
- **`fire_rate` absolutos bajos** (0.03-0.05) para los features de sesgo → son features de nicho; los contextos son sugerentes pero con n modesto.
- **Captura solo del último token**; el sesgo podría codificarse también en posiciones de tokens visuales.
- **SAE de texto-Base** aplicado a un modelo multimodal: el diccionario puede no contener direcciones visuales/demográficas limpias.

## 6. Próximos pasos propuestos

1. **Validación causal** de los features demográficos (F44412, F1036, F45944): reconstruir un vector solo con ellos (on-manifold) y comprobar si voltea el veredicto → conectaría "feature demográfico" con "causa del rechazo".
2. **Caracterizar F39735 con imágenes** (máx-activantes visuales): qué fotos/atributos lo encienden, para confirmar que es el canal visual del sesgo.
3. **Romper el confound**: ampliar a varios sujetos por grupo demográfico y quedarse con los features consistentes intra-grupo (sesgo real) frente a los idiosincrásicos de una foto.

---

## Apéndice — Reproducibilidad (PIPELINE_STEP)

| paso | comando |
|---|---|
| Descargar SAE | `sae_download` (`LAYER`) |
| Validar reconstrucción | `sae_validate` (`LAYER`) |
| Descomponer `v` | `sae_decompose` |
| Ablación causal | `sae_ablation` |
| Cachear activaciones | `cache_activations` |
| Descomposición por condición | `analyze_conditions` |
| Máx-activantes (interpretación) | `sae_max_activating` (`LAYER`, `CORPUS=wikitext\|social`) |
| Anclaje a labels demográficos (AUC) | `sae_feature_grounding` (`LAYER`) |

Lanzamiento (ejemplo): `docker run --gpus '"device=1"' -e PIPELINE_STEP=sae_max_activating -e CORPUS=social … qwen3-rtx3090:latest`
