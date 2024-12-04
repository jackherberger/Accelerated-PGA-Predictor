package FinalProject

import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection._
import com.opencsv.CSVParserBuilder
import com.opencsv.CSVReaderBuilder
import java.io.StringReader
import scala.math.pow



object FinalProject {

  case class Round(
     name: String, dk: Double, fd: Double, purse: Double, course: String, par: Double, yardage: Double,
     puttSg: Double, argSg: Double, appSg: Double, ottSg: Double, strokesPerRound: Double) {

    override def toString: String = {
      s"Round(name: $name, dk: $dk, fd: $fd, purse: $purse, course: $course, par: $par, " +
      s"yardage: $yardage, puttSg: $puttSg, argSg: $argSg, appSg: $appSg, ottSg: $ottSg, " +
      s"strokesPerRound: $strokesPerRound)"
    }
  }

  class KNN(k: Int) extends Serializable {
    private var trainingData: RDD[Round] = _
    def fit(data: RDD[Round]): Unit = {
      trainingData = data
    }
    def EuclideanDistance(round1: Round, round2: Round): Double = {
      pow(pow(round1.dk - round2.dk, 2) +
        pow(round1.fd - round2.fd, 2) +
        pow(round1.purse - round2.purse, 2) +
        pow(round1.par - round2.par, 2) +
        pow(round1.yardage - round2.yardage, 2) +
        pow(round1.puttSg - round2.puttSg, 2) +
        pow(round1.appSg - round2.dk, 2) +
        pow(round1.ottSg - round2.dk, 2), 0.5)
    }
    def predict(r1: Round): Double = {
      // Calculate the distances to all training points
      val distances = trainingData.map { r2 =>
        val distance = EuclideanDistance(r1, r2)
        (r1.name, distance, r2.strokesPerRound)
      }
      // get the k nearest neighbors
      val neighbors = distances.sortBy(_._2).collect().slice(1, k+1)
      // return the average of stroke count for k neighbors
      neighbors
        .map(_._3)
        .sum / k
    }
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    // spark context creation
    val conf = new SparkConf().setAppName("NameOfApp").setMaster("local[4]");
    val sc = new SparkContext(conf);

    val courses = sc.textFile("src/main/dg_course_table.csv")
    val players = sc.textFile("src/main/PGA_Tourn_Level_Data.csv")

    val cleanedCourses = courses.filter(line => line != null && line.nonEmpty)
    val cleanedPlayers = players.filter(line => line != null && line.nonEmpty)

    val data = getData(cleanedCourses, cleanedPlayers)

    KNearestNeighbors(StandardScaler(data), 10, 10)

    Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
      .foreach(k => KNearestNeighbors(StandardScaler(data), k, 10, 42))


    //KNearestNeighbors(data, 10, 10)

  }

  def StandardScaler(data: RDD[Round]): RDD[Round] = {
    val stats = data.map { round =>
      (
        round.dk, round.fd, round.purse, round.par, round.yardage, round.puttSg,
        round.argSg, round.appSg, round.ottSg, round.strokesPerRound
      )
    }.mapPartitions { iter =>
      val stats = iter.foldLeft((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0L)) {
        case ((sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, count),
        (dk, fd, purse, par, yardage, puttSg, argSg, appSg, ottSg, strokes)) =>
          (
            sum1 + dk, sum2 + fd, sum3 + purse, sum4 + par, sum5 + yardage,
            sum6 + puttSg, sum7 + argSg, sum8 + appSg, sum9 + ottSg, sum10 + strokes,
            count + 1
          )
      }
      Iterator(stats)
    }.reduce { (a, b) =>
      (
        a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4, a._5 + b._5,
        a._6 + b._6, a._7 + b._7, a._8 + b._8, a._9 + b._9, a._10 + b._10,
        a._11 + b._11
      )
    }
    val (sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, count) = stats
    val means = (
      sum1 / count, sum2 / count, sum3 / count, sum4 / count, sum5 / count,
      sum6 / count, sum7 / count, sum8 / count, sum9 / count, sum10 / count
    )

    val variances = data.map { round =>
      val diffs = (
        round.dk - means._1, round.fd - means._2, round.purse - means._3, round.par - means._4,
        round.yardage - means._5, round.puttSg - means._6, round.argSg - means._7,
        round.appSg - means._8, round.ottSg - means._9, round.strokesPerRound - means._10
      )
      (
        diffs._1 * diffs._1, diffs._2 * diffs._2, diffs._3 * diffs._3, diffs._4 * diffs._4,
        diffs._5 * diffs._5, diffs._6 * diffs._6, diffs._7 * diffs._7, diffs._8 * diffs._8,
        diffs._9 * diffs._9, diffs._10 * diffs._10
      )
    }.reduce { (a, b) =>
      (
        a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4, a._5 + b._5,
        a._6 + b._6, a._7 + b._7, a._8 + b._8, a._9 + b._9, a._10 + b._10
      )
    }

    val stdDevs = (
      Math.sqrt(variances._1 / count), Math.sqrt(variances._2 / count),
      Math.sqrt(variances._3 / count), Math.sqrt(variances._4 / count),
      Math.sqrt(variances._5 / count), Math.sqrt(variances._6 / count),
      Math.sqrt(variances._7 / count), Math.sqrt(variances._8 / count),
      Math.sqrt(variances._9 / count), Math.sqrt(variances._10 / count)
    )

    data.map { round =>
      Round(
        name = round.name,
        dk = (round.dk - means._1) / stdDevs._1,
        fd = (round.fd - means._2) / stdDevs._2,
        purse = (round.purse - means._3) / stdDevs._3,
        course = round.course,
        par = (round.par - means._4) / stdDevs._4,
        yardage = (round.yardage - means._5) / stdDevs._5,
        puttSg = (round.puttSg - means._6) / stdDevs._6,
        argSg = (round.argSg - means._7) / stdDevs._7,
        appSg = (round.appSg - means._8) / stdDevs._8,
        ottSg = (round.ottSg - means._9) / stdDevs._9,
        strokesPerRound = round.strokesPerRound
      )
    }
  }

  def KNearestNeighbors(data: RDD[Round], k: Int, test_cnt: Int, seed: Int): Unit = {
    val model = new KNN(k)
    model.fit(data)

    val test = data.takeSample(withReplacement = false, num = test_cnt, seed = 42)

    val predictionsAndActuals = test.map(round => {
      val predicted = model.predict(round)
      val actual = round.strokesPerRound
      println(s"${round.name} on ${round.course} --- Predicted: $predicted, Actual: $actual")
      (predicted, actual)
    })

    val metrics = predictionsAndActuals.map { case (predicted, actual) =>
      val error = predicted - actual
      (math.abs(error), error * error, predicted, actual)
    }.reduce { (a, b) =>
      (
        a._1 + b._1, // Sum of absolute errors
        a._2 + b._2, // Sum of squared errors
        a._3 + b._3, // Sum of predictions
        a._4 + b._4 // Sum of actual values
      )
    }

    val mae = metrics._1 / test.length // Mean Absolute Error
    val mse = metrics._2 / test.length // Mean Squared Error
    val rmse = math.sqrt(mse) // Root mean squared error

    val sumActual = metrics._4
    val sse = metrics._2
    val n = test.length // Total number of observations
    val meanActual = sumActual / n

    // Calculate TSS
    val tss = predictionsAndActuals.map { case (actual, _) =>
      math.pow(actual - meanActual, 2) }.sum

    // Calculate R^2
    val r2 = 1 - (sse / tss)

    println(f"Metrics:")
    println(f"MAE: $mae%.2f")
    println(f"MSE: $mse%.2f")
    println(f"RMSE: $rmse%.2f")
    println(f"R^2: $r2%.2f")

  }

  def getData(courses: RDD[String], players: RDD[String]): RDD[Round] = {
    // (course-name, (par, yardage, putt_sg, arg_sg, app_sg, ott_sg))
    val coursesDF = courses
      .map { line =>
        val parser = new CSVParserBuilder().withSeparator(',').withQuoteChar('"').build()
        val reader = new CSVReaderBuilder(new StringReader(line)).withCSVParser(parser).build()
        reader.readNext()
      }
      .map(fields =>
        (fields(0).toUpperCase.trim,
          (fields(1).toInt, fields(2).toDouble, fields(12).toDouble, fields(13).toDouble, fields(14).toDouble, fields(15).toDouble))
      )

    // (course-name, (name, dk, fd, purse, n_rounds, strokes)
    val playersDF = players
      .map { line =>
        val parser = new CSVParserBuilder().withSeparator(',').withQuoteChar('"').build()
        val reader = new CSVReaderBuilder(new StringReader(line)).withCSVParser(parser).build()
        reader.readNext()
      }
      .map(fields =>
        (fields(25).split("-")(0).toUpperCase.trim,
          (fields(20).trim, fields(14).toDouble, fields(15).toDouble, fields(27).toDouble, fields(11).toInt, fields(4).toDouble))
      )

    // (name, dk, fd, purse, course, par, yardage, putt_sg, arg_sg, app_sg, ott_sg, strokes / n_rounds)
    playersDF
      .join(coursesDF)
      .map({ case (course, ((name, dk, fd, purse, n_rounds, strokes), (par, yardage, putt_sg, arg_sg, app_sg, ott_sg))) =>
        Round(name, dk, fd, purse, course, par, yardage, putt_sg, arg_sg, app_sg, ott_sg, strokes / n_rounds)
      })
  }


}