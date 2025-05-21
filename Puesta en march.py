from pyspark.sql import SparkSession
import os

# Configuración del archivo de entrada
input_file = r"C:\Users\Ricardo\PycharmProjects\ArtifialInteligence\ejemplo.csv"

# Verificación previa del archivo
if not os.path.exists(input_file):
    print(f"❌ Archivo no encontrado: {input_file}")
    # Crear archivo de prueba si no existe
    with open(input_file, "w", encoding="utf-8") as f:
        f.write("Nombre,Edad\nRicardo,25\nAna,30\nLuis,22\n")
    print(f"📁 Archivo de prueba creado en: {input_file}")

# Crear SparkSession
spark = SparkSession.builder \
    .appName("Prueba PySpark Básica") \
    .master("local[*]") \
    .getOrCreate()

print("✅ SparkSession iniciada")

# Leer CSV con encabezado
df = spark.read.csv(input_file, header=True, inferSchema=True)

# Mostrar contenido
print("📊 Contenido del DataFrame:")
df.show()

# Operación: promedio de edad
df.selectExpr("avg(Edad) as Promedio_Edad").show()

# Finalizar Spark
spark.stop()
print("🛑 SparkSession detenida correctamente")
