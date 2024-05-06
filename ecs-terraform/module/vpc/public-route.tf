resource "aws_route_table" "public-route-table" {
  vpc_id = aws_vpc.vpc.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = {
    Name  = "Public-route"
    from  = "trrafrom"
    owner = "manoj"
  }
}

resource "aws_route_table_association" "association" {
  route_table_id = aws_route_table.public-route-table.id
  count          = length(var.public-cidr)
  subnet_id      = aws_subnet.public-subnet-test[count.index].id
}