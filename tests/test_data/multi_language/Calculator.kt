package demo

/** Calculator math contract. */
interface MathOperations {
    fun add(a: Int, b: Int): Int
}

enum class Operation {
    ADD,
    SUBTRACT,
}

/**
 * A simple calculator.
 */
@Suppress("unused")
data class Calculator(val name: String) : MathOperations {
    companion object {
        fun create(): Calculator = Calculator("generated")
    }

    init {
        require(name.isNotBlank())
    }

    constructor() : this("default")

    /** Adds two numbers. */
    override fun add(a: Int, b: Int): Int {
        return a + b
    }

    val version: String = "1.0"
}

/** Convert a string to a Calculator instance. */
fun String.toCalculator(): Calculator = Calculator(this)

