def eliminar_primera_ultima_columna(archivo_entrada, archivo_salida):
    with open(archivo_entrada, "r", encoding="utf-8") as f_in, open(archivo_salida, "w", encoding="utf-8") as f_out:
        for _ in range(5):
            next(f_in, None)
        for linea in f_in:
            columnas = linea.strip().split(",")  # Divide la línea en columnas usando ","
            if len(columnas) > 2:  # Si hay más de dos columnas, elimina la primera y última
                nueva_linea = ",".join(columnas[1:-1])
                f_out.write(nueva_linea + "\n")
            elif len(columnas) == 2:  # Si solo hay dos columnas, escribe una línea vacía
                f_out.write("\n")
            else:  # Si hay solo una columna o está vacío, mantiene la línea vacía
                f_out.write("\n")

# Configuración del programa
nombre="Nombre_del_archivo_de_spectrum_studio"
ext=".txt"
x="X"
for i in range(2,16):
    archivo_entrada = nombre+str(i)+ext  # Nombre del archivo de entrada
    archivo_salida = nombre+str(i)+x+ext  # Archivo donde se guardará el resultado
    eliminar_primera_ultima_columna(archivo_entrada, archivo_salida)

print(f"Proceso completado. Se ha generado '{archivo_salida}'.")

