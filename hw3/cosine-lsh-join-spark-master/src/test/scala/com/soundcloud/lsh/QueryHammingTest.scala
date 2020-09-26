package com.soundcloud.lsh

import com.soundcloud.TestHelper
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.scalatest.{FunSuite, Matchers}

class QueryHammingTest
  extends FunSuite
    with SparkLocalContext
    with Matchers
    with TestHelper {

  def denseVector(input: Double*): Vector = {
    Vectors.dense(input.toArray)
  }

  val queryVectorA = denseVector(1.0, 1.0)
  val queryVectorB = denseVector(-1.0, 1.0)

  val catalogVectorA = denseVector(1.0, 1.0)
  val catalogVectorB = denseVector(-1.0, 1.0)
  val catalogVectorC = denseVector(-1.0, 0.5)
  val catalogVectorD = denseVector(1.0, 0.5)

  val queryRows = Seq(
    IndexedRow(0, queryVectorA),
    IndexedRow(1, queryVectorB)
  )

  val catalogRows = Seq(
    IndexedRow(0, catalogVectorA),
    IndexedRow(1, catalogVectorB),
    IndexedRow(2, catalogVectorC),
    IndexedRow(3, catalogVectorD)
  )

  val expected = Array(
    MatrixEntry(0, 0, Cosine(queryVectorA, catalogVectorA)),
    MatrixEntry(0, 3, Cosine(queryVectorA, catalogVectorD)),
    MatrixEntry(1, 1, Cosine(queryVectorB, catalogVectorB)),
    MatrixEntry(1, 2, Cosine(queryVectorB, catalogVectorC))
  )

  test("broadcast catalog") {
    val queryMatrix = new IndexedRowMatrix(sc.parallelize(queryRows))
    val catalogMatrix = new IndexedRowMatrix(sc.parallelize(catalogRows))

    val queryNearestNeighbour = new QueryHamming(0.1, 10000, 2, true)
    val got = queryNearestNeighbour.join(queryMatrix, catalogMatrix).entries.collect

    implicit val equality = new MatrixEquality(0.02)
    got.sortBy(t => (t.i, t.j)) should equal(expected)
  }

  test("broadcast query") {
    val queryMatrix = new IndexedRowMatrix(sc.parallelize(queryRows))
    val catalogMatrix = new IndexedRowMatrix(sc.parallelize(catalogRows))

    val queryNearestNeighbour = new QueryHamming(0.1, 10000, 2, false)
    val got = queryNearestNeighbour.join(queryMatrix, catalogMatrix).entries.collect

    implicit val equality = new MatrixEquality(0.02)
    got.sortBy(t => (t.i, t.j)) should equal(expected)
  }

}


