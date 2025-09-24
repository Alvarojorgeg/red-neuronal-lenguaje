# Preguntas de análisis

## 1. Complejidad de implementar backpropagation a mano

- **Complejidad y errores**. Derivar y codificar gradientes para cada capa es proclive a errores de signo y dimensiones.
- **Estabilidad numérica**. Sin ayudas del framework habría que lidiar manualmente con gradientes que se desvanecen o explotan.
- **Eficiencia**. Optimizar la vectorización, el batching y la reutilización de memoria resulta difícil sin kernels optimizados.
- **Modularidad**. Añadir nuevas activaciones o funciones de pérdida implicaría re-derivar y probar todas las fórmulas.
- **Depuración**. Sin herramientas de autograd, detectar el origen de NaNs o gradientes nulos es muy costoso.
- **Hardware**. Sacar provecho de GPU/TPU manualmente supone diseñar kernels específicos.

## 2. Ventajas del intérprete como capa de abstracción

- **Productividad**. Probar variantes de arquitecturas es tan simple como editar una cadena de texto versionable.
- **Legibilidad**. El mini-lenguaje concentra la arquitectura en una línea fácil de revisar en equipo.
- **Portabilidad**. El mismo intérprete puede apuntar a backends distintos (Keras, PyTorch, JAX) sin cambiar la sintaxis.
- **Validación temprana**. El parser detecta errores de configuración antes de lanzar un entrenamiento costoso.
- **Automatización**. Facilita generar configuraciones automáticamente para búsquedas de hiperparámetros.
