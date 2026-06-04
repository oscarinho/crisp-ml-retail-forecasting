# 🍋 Explicado como si tuvieras 10 años

> Archivo local — no se publica. Es la versión más simple del proyecto,
> útil para explicárselo a alguien que no sabe nada de tecnología, a
> un familiar, o a uno mismo cuando uno se pierde en la complejidad.

---

## El puesto de limonada

Imagina que tienes un puesto de limonada en frente de tu casa.

Todos los días tienes que decidir cuántas limonadas preparar.

🥤 Si preparas **muy pocas** → vienen niños del barrio, no hay para
todos, se van enojados.

🥤 Si preparas **muchas** → te sobran, se calientan, las botas. Perdiste plata.

La pregunta más difícil del día es:

> *"¿Cuánta limonada voy a vender mañana?"*

Esa misma pregunta se la hacen los supermercados, las tiendas de ropa,
las farmacias, y cualquier negocio que vende cosas. Solo que en vez de
un puesto, tienen **5.000 productos en 50 tiendas**. Y se equivocan
todos los días.

---

## ¿Qué construí?

Construí un **asistente** que mira cómo se vendió antes y dice:

> *"Mañana va a venir más gente porque hace calor, hay feriado y la
> tienda de al lado subió los precios. Prepárate para vender más."*

O al revés:

> *"Mañana es un día tranquilo. Haz menos limonada, así no la botas."*

---

## ¿Cómo lo hace?

Mira muchas cosas a la vez:

- 📅 Qué día de la semana es (los sábados se vende más)
- ☀️ Si hace frío o calor
- 🎉 Si hay un partido importante o un feriado
- 🏪 Cuánto cobra el puesto de al lado
- 📈 Cuánto vendiste la semana pasada

Y junta toda esa información en **un número**: "mañana vendes tantas".

---

## ¿Qué aprendí en el camino?

### 1. Hay datos que parecen útiles y son trampa

Había un dato que decía *"el número exacto de limonadas que vas a
vender mañana"*. Si lo usaba, mi asistente parecía un genio.

Pero el día de hacer la limonada en serio, ese dato **no existe
todavía**. Era trampa.

> **Lo que aprendí:** solo sirve lo que vas a tener cuando llegue el
> momento real de decidir. Lo demás suena lindo y rompe.

### 2. A veces lo simple le gana a lo complicado

Probé un asistente súper completo con **29 datos** a la vez. Y otro
más simple con solo **15 datos**. El simple ganó.

¿Por qué? Porque el complicado **memorizaba** en vez de aprender.

Es como cuando estudias para un examen aprendiendo las respuestas de
memoria — sacas 10 ese día, pero al siguiente examen no sabes nada.

> **Lo que aprendí:** más información no siempre es mejor decisión.

### 3. Saber cuánto vas a vender NO te dice cuánto preparar

Aunque suene raro, son dos cosas distintas.

Si tu asistente dice "mañana se venden 100 limonadas", tú **no quieres
preparar exactamente 100**. Quieres:

- Tener un poquito más, por si vienen más niños. 🙋
- Pero no MUCHO más, porque si no, botas. 🗑️

Cuánto exactamente "un poquito más" es **otra decisión** que depende
de:

- Cuánto tarda la limonada en hacerse (si tu mamá tarda 2 horas, no
  puedes esperar al último momento)
- Qué tan importante es no quedarte corto (¿es solo limonada o es algo
  que la gente necesita sí o sí?)
- Cuánto te duele botar vs. cuánto te duele decepcionar al cliente

> **Lo que aprendí:** el pronóstico es solo la mitad del trabajo.
> La otra mitad es decidir qué haces con ese pronóstico.

### 4. No todas las limonadas se planifican igual

Si tu puesto vende:

- 🍋 **Limonada clásica** — todos los días, todos la piden → fácil de
  planear, tu asistente lo hace solo.
- 🍓 **Limonada de fresa** — a veces sí, a veces no → más difícil,
  necesita que tú le prestes atención cada semana.
- 🥝 **Limonada de kiwi** — una vez al mes alguien la pide → ni te
  molestes en hacer un sistema. Cuando alguien la pida, la haces.

> **Lo que aprendí:** poner el mismo esfuerzo en cada producto es la
> forma más rápida de perder tiempo en lo que no importa.

---

## ¿Para qué sirve todo esto?

Las tiendas pierden **mucha plata** cuando:

- 😢 No tienen lo que el cliente quiere → el cliente se va a otra
  tienda y, peor, no vuelve más.
- 😢 Tienen demasiado → se queda en la estantería juntando polvo, y es
  plata atrapada que no rota.

Mi asistente ayuda a no equivocarse tanto.

Si una cadena de tiendas mejora un poquito sus pronósticos, puede
ahorrar **entre 1 y 3 millones de dólares al año**.

No es un número inventado — es lo que dicen las consultoras que miden
estas cosas (Gartner, McKinsey).

---

## ¿Y por qué cuento todo esto en internet?

Por tres razones:

1. La mayoría de los gerentes de tiendas **no saben** que esto se puede
   hacer.
2. Si publico lo que aprendí, gente que tiene el mismo problema me
   puede encontrar.
3. Cuando uno explica lo que hace en palabras simples, se entiende
   mejor a sí mismo. Esta página es la prueba.

---

## Si te preguntas "¿cómo funciona por dentro?"

Eso ya es ciencia de datos:

- Hay matemática 🔢
- Hay programación 💻
- Hay decisiones técnicas que se discuten entre profesionales 🤓

Pero **todo lo importante para entender por qué importa** cabe en
este texto.

El resto vive en [oscarponce.com](https://oscarponce.com/labs).

---

## Mini-diccionario por si escuchas a un grande hablar

| Palabra grande | Significa | En el ejemplo del puesto |
|---|---|---|
| Forecast / Pronóstico | Adivinar cuánto vas a vender | "Mañana 100 limonadas" |
| Demanda | Cuántos quieren comprar | Los niños que pasaron por el puesto |
| Stockout | No tener cuando alguien quiere comprar | Te quedaste sin limones |
| Overstock | Tener de más | Te sobraron 30 limonadas, las botas |
| Inventario | Lo que tienes en stock | Las limonadas en la nevera |
| Lead time | Cuánto tarda en llegar lo que pediste | Lo que tarda tu mamá en comprar limones |
| Safety stock | El colchón extra "por si acaso" | 10 limonadas de más, por si vienen amigos |
| Service level | Qué tan seguido logras tener stock | "Casi siempre tengo limonada" = 95% |
| Modelo / Algoritmo | El asistente que mira los números | El "cerebro" del puesto |
| Machine Learning | Cuando el asistente aprende solo de los datos | "Cada día aprende un poquito más" |
| Leakage | Hacer trampa con los datos | Usar info que mañana no vas a tener |

---

**Si llegaste hasta aquí:** ya entiendes el 80% del proyecto.

El 20% restante es código y matemática, y vive en los notebooks. Pero
nadie debería sentirse afuera de esta conversación solo porque no sabe
Python — el problema (vender bien, no perder plata, no decepcionar
clientes) lo entiende cualquiera.

Y resolverlo bien es valioso, no importa si lo haces con un asistente
de inteligencia artificial o con una libreta y un lápiz.

Lo importante es que la pregunta esté bien hecha.
