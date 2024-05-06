# resource "aws_instance" "jump-server" {
#   ami = "ami-007020fd9c84e18c7"
#   instance_type = "t2.micro"
#   subnet_id = aws_subnet.public-subnet[0].id
#   key_name = "ksv"
# }

# resource "aws_instance" "private-server" {
#   ami = "ami-007020fd9c84e18c7"
#   instance_type = "t2.micro"
#   subnet_id = aws_subnet.private-subnet[0].id
#   key_name = "ksv"
# }