output "public-subnet" {
  value = aws_subnet.public-subnet-test[*].id
}

output "private-subnet" {
  value = aws_subnet.private-subnet[*].id
}

output "vpc" {
  value = aws_vpc.vpc.id
}