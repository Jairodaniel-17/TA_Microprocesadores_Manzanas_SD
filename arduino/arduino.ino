const int pin13 = 13;
const int pin12 = 12;
void setup()
{
    pinMode(pin13, OUTPUT);
    pinMode(pin12, OUTPUT);
    Serial.begin(9600);
}

void loop()
{
    if (Serial.available() > 0)
    {
        char c = Serial.read();
        if (c == 'b')
        {
            digitalWrite(pin13, HIGH);
            digitalWrite(pin12, LOW);
        }
        if (c == 'm')
        {
            digitalWrite(pin13, LOW);
            digitalWrite(pin12, HIGH);
        }
    }
}