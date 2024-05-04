resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.vpc.id
  tags = {
    Name  = "DEV-vpd_IGW"
    from  = "trrafrom"
    owner = "manoj"
  }
}